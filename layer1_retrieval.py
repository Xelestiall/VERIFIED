"""
LAYER 1: Memastikan jawaban berbasis dokumen aktual, bukan parametric memory. 
PDF -> chunk -> embedding -> vector store.

Stack:
  LangChain  -> loader, splitter, vector store (sesuai Bab 3 tesis)
  Voyage AI  -> embedding (voyage-finance-2, domain keuangan).
  FAISS      -> vector store lokal, gratis, bisa di-persist ke disk
  NetworkX   -> knowledge graph entitas (opsional, buat multi-hop)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from config import RETRIEVAL, get_api_key

# ----------------------------------------------------------
# Struktur data internal
@dataclass
class Chunk:
    """
    Satu potongan dokumen + semua metadata yang dibutuhkan layer atas.

    `tags` diisi Layer 2, `score` diisi saat retrieval.
    Dibikin satu dataclass biar nggak ada dict liar berkeliaran
    antar layer — ini yang bikin kode gampang di-debug pas run panjang.
    """
    chunk_id: str
    text: str
    source: str = ""
    page: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    score: float = 0.0

    def citation(self) -> str:
        """Label sitasi yang muncul di jawaban -> bukti traceability (H4)."""
        page = f"hal. {self.page}" if self.page is not None else "hal. ?"
        return f"[{self.chunk_id} | {os.path.basename(self.source)} | {page}]"


# ----------------------------------------------------------
# Embedding adapter
# ----------------------------------------------------------
class VoyageEmbeddings(Embeddings):
    """
    Adapter Voyage AI yang kompatibel dengan interface Embeddings LangChain.

    input_type="document" vs "query" itu bukan kosmetik — Voyage melatih
    dua proyeksi berbeda, dan salah pasang bikin recall turun.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import voyageai

        key = get_api_key("VOYAGE_API_KEY", api_key or "")
        if not key:
            raise ValueError(
                "VOYAGE_API_KEY kosong. Daftar di dashboard.voyageai.com, lalu "
                "salin .env.example jadi .env dan isi key-nya, atau:\n"
                "  export VOYAGE_API_KEY='pa-...'"
            )
        self.client = voyageai.Client(api_key=key)
        self.model = model or RETRIEVAL.embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        bs = RETRIEVAL.embedding_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            r = self.client.embed(batch, model=self.model, input_type="document")
            out.extend(r.embeddings)
        return out

    def embed_query(self, text: str) -> List[float]:
        r = self.client.embed([text], model=self.model, input_type="query")
        return r.embeddings[0]


class LocalEmbeddings(Embeddings):
    """
    Fallback gratis (sentence-transformers, jalan offline).

    Berguna buat: (a) debugging pipeline tanpa bakar kredit,
    (b) ablation "apakah embedding domain-finance benar-benar berpengaruh?"
    Itu satu paragraf temuan gratis di Bab 4 lo.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [f"query: {text}"], normalize_embeddings=True
        )[0].tolist()


def get_embeddings(provider: str = "voyage"):
    """Factory. Ganti string di sini = ganti seluruh strategi embedding."""
    if provider == "voyage":
        return VoyageEmbeddings()
    if provider == "local":
        return LocalEmbeddings()
    raise ValueError(f"Provider embedding tidak dikenal: {provider}")


# ----------------------------------------------------------
# Retriever
# ----------------------------------------------------------
class RetrievalLayer:
    """
    Bungkus ingest + search jadi satu objek per-dokumen.

    Pola pakai:
        r = RetrievalLayer("data/pdf/BBCA_2025.pdf")
        r.build()              # sekali; index disimpan ke disk
        chunks = r.search("arus kas operasi", top_k=8)
    """

    def __init__(self, pdf_path: str, embedding_provider: str = "voyage",
                 index_dir: Optional[str] = None):
        self.pdf_path = pdf_path
        self.doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.embeddings = get_embeddings(embedding_provider)
        self.vectorstore = None
        self.chunks: List[Chunk] = []
        # Default: RETRIEVAL.index_dir (dipakai runner.py, cache batch yg
        # sengaja persisten). app.py Streamlit meng-override ini dgn folder
        # temp per-session supaya index tidak numpuk permanen di server.
        self.index_dir = index_dir or RETRIEVAL.index_dir

    # ---------- ingest ----------
    def load_and_split(self) -> List[Chunk]:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = PyPDFLoader(self.pdf_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RETRIEVAL.chunk_size,
            chunk_overlap=RETRIEVAL.chunk_overlap,
            separators=RETRIEVAL.separators,
            length_function=len,
        )
        splits = splitter.split_documents(docs)

        self.chunks = [
            Chunk(
                chunk_id=f"{self.doc_id}::c{i:04d}",
                text=d.page_content,
                source=self.pdf_path,
                # PyPDFLoader pakai index 0; +1 biar cocok sama nomor
                # halaman yang dilihat manusia saat verifikasi manual.
                page=d.metadata.get("page", 0) + 1,
            )
            for i, d in enumerate(splits)
        ]
        return self.chunks

    def build(self, force_rebuild: bool = False) -> "RetrievalLayer":
        """Bangun index. Kalau sudah ada di disk, load saja (hemat kredit)."""
        from langchain_community.vectorstores import FAISS

        index_path = os.path.join(self.index_dir, self.doc_id)
        meta_path = index_path + "_chunks.pkl"

        if not force_rebuild and os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            self.vectorstore = FAISS.load_local(
                index_path, self.embeddings, allow_dangerous_deserialization=True
            )
            return self

        self.load_and_split()
        os.makedirs(self.index_dir, exist_ok=True)

        self.vectorstore = FAISS.from_texts(
            texts=[c.text for c in self.chunks],
            embedding=self.embeddings,
            metadatas=[
                {"chunk_id": c.chunk_id, "page": c.page, "source": c.source}
                for c in self.chunks
            ],
        )
        self.vectorstore.save_local(index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        return self

    def table_scales(self) -> set:
        """Skala satuan tabel yang ditemukan di SELURUH dokumen (mis. {1e6}).

        Dipakai gate numerik (Layer 3) karena header 'dalam jutaan Rupiah'
        tidak selalu ikut terambil bersama chunk badan tabelnya.
        """
        if not hasattr(self, "_table_scales"):
            from layer3_verification import detect_table_scale  # impor lazy: hindari siklus
            self._table_scales = {detect_table_scale(c.text) for c in self.chunks} - {1.0}
        return self._table_scales

    # ---------- search ----------
    def search(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """
        Ambil chunk paling relevan.

        FAISS mengembalikan L2 distance (kecil = mirip). Dikonversi ke
        similarity 0..1 supaya bisa dipadukan dengan skor tag di Layer 2.
        """
        if self.vectorstore is None:
            raise RuntimeError("Index belum dibangun. Panggil .build() dulu.")

        k = top_k or RETRIEVAL.top_k
        hits = self.vectorstore.similarity_search_with_score(query, k=k)

        by_id = {c.chunk_id: c for c in self.chunks}
        out: List[Chunk] = []
        for doc, dist in hits:
            cid = doc.metadata.get("chunk_id")
            base = by_id.get(cid)
            c = Chunk(
                chunk_id=cid or "unknown",
                text=doc.page_content,
                source=doc.metadata.get("source", self.pdf_path),
                page=doc.metadata.get("page"),
                tags=list(base.tags) if base else [],
            )
            c.score = 1.0 / (1.0 + float(dist))
            out.append(c)
        return out


# ----------------------------------------------------------
# Knowledge graph (opsional)
# ----------------------------------------------------------
class KnowledgeGraph:
    """
    Graf entitas in-memory pakai NetworkX (Peng et al., 2024).

    Kapan ini berguna: pertanyaan multi-hop seperti Q6 ("bisnis apa saja
    di bawah perusahaan ini") — jawabannya sering tersebar di beberapa
    halaman, dan vector search sendirian gampang miss.
    """

    def __init__(self):
        import networkx as nx

        self.g = nx.DiGraph()

    def add_triples(self, triples: List[Dict[str, str]], source_chunk: str = "") -> None:
        for t in triples:
            s, r, o = t.get("subject"), t.get("relation"), t.get("object")
            if not (s and r and o):
                continue
            self.g.add_node(s, type=t.get("subject_type", "entity"))
            self.g.add_node(o, type=t.get("object_type", "entity"))
            self.g.add_edge(s, o, relation=r, source=source_chunk)

    def extract_from_chunks(self, chunks: List[Chunk], llm: Any, max_chunks: int = 15) -> None:
        """
        Ekstraksi triple pakai model murah (utility_model).

        Dibatasi max_chunks karena ini O(n) call ke API — 200 chunk x 30
        emiten bakal bikin tagihan lo meledak tanpa nambah nilai riset.
        """
        from config import MODEL

        system = (
            "Anda adalah ekstraktor entitas laporan keuangan Indonesia. "
            "Ambil relasi antar entitas dalam bentuk triple."
        )
        for c in chunks[:max_chunks]:
            prompt = (
                "Ekstrak maksimal 5 triple (subject, relation, object) dari teks berikut. "
                'Format: {"triples":[{"subject":"...","relation":"...","object":"..."}]}\n'
                "Fokus pada: entitas anak perusahaan, segmen usaha, kepemilikan, lini bisnis.\n"
                "Kalau tidak ada relasi jelas, kembalikan list kosong.\n\n"
                f"TEKS:\n{c.text[:1500]}"
            )
            res = llm.complete_json(
                prompt,
                system=system,
                model=MODEL.utility_model,
                stage="kg_extraction",
                fallback={"triples": []},
            )
            self.add_triples(res.get("triples", []), source_chunk=c.chunk_id)

    def neighbors_text(self, entity: str, hops: int = 1) -> str:
        """Ratakan tetangga entitas jadi teks, siap ditempel ke context."""
        import networkx as nx

        if entity not in self.g:
            return ""
        nodes = nx.single_source_shortest_path_length(
            self.g.to_undirected(), entity, cutoff=hops
        )
        lines = [
            f"- {u} --[{d.get('relation', 'terkait')}]--> {v}"
            for u, v, d in self.g.edges(data=True)
            if u in nodes or v in nodes
        ]
        return "\n".join(lines[:30])

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}
