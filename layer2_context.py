"""
layer2_context.py — LAYER 2: CONTEXT MANAGEMENT

Ini pembeda utama VERIFIED vs Standard RAG. Standard RAG cuma
menjejalkan top-k chunk apa adanya. Layer ini melakukan 4 hal:

  1. TAGGING     — tandai tiap chunk pakai regex (REGEX_TAGS di config)
  2. RERANKING   — skor gabungan: similarity + kecocokan tag dengan query
  3. BUDGETING   — potong sampai muat budget token (lawan context bloat)
  4. REORDERING  — taruh chunk terbaik di awal & akhir (lawan
                   lost-in-the-middle, Liu et al., 2024)

Tiap fungsi bisa dimatikan satu-satu lewat config -> ini yang bikin
lo bisa bikin tabel ablation study yang rapi di Bab 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import CONTEXT, REGEX_TAGS
from layer1_retrieval import Chunk


# Compile sekali di import. Compile per-panggilan pada 200 chunk x 8
# pertanyaan x 30 emiten = pemborosan CPU yang nggak perlu.
COMPILED_TAGS: Dict[str, re.Pattern] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in REGEX_TAGS.items()
}


# ----------------------------------------------------------
# 1. Tagging
# ----------------------------------------------------------
def tag_text(text: str) -> List[str]:
    """Kembalikan daftar tag yang cocok pada sepotong teks."""
    return [name for name, rx in COMPILED_TAGS.items() if rx.search(text)]


def tag_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Tandai chunk in-place lalu kembalikan (biar bisa di-chain)."""
    for c in chunks:
        c.tags = tag_text(c.text)
    return chunks


def tag_query(query: str) -> List[str]:
    """
    Tag untuk pertanyaan.

    Query itu pendek, jadi regex sering nggak nyantol. Karena itu ada
    lapisan kedua: kata kunci intent -> tag. Contoh: kata "berapa"
    sinyal bahwa jawabannya HARUS mengandung angka, jadi chunk
    ber-tag CURRENCY/NUMBER pantas diprioritaskan.
    """
    tags = set(tag_text(query))

    intent_map = {
        r"berapa|nominal|jumlah|nilai|total": ["CURRENCY", "NUMBER"],
        r"persen|persentase|rasio|pertumbuhan": ["PERCENTAGE"],
        r"arus\s+kas|cash\s*flow|kas": ["CASHFLOW"],
        r"dividen": ["DIVIDEND"],
        r"laba|rugi|profit": ["PROFIT"],
        r"segmen|divisi|lini\s+bisnis|anak\s+perusahaan": ["SEGMENT"],
        r"kurs|nilai\s+tukar|valas|rupiah|usd": ["FX_RISK"],
        r"energi|bahan\s+bakar|listrik": ["ENERGY_COST"],
        r"esg|keberlanjutan|emisi|lingkungan": ["ESG"],
        r"mitigasi|strategi|risiko|lindung\s+nilai": ["RISK_MITIGATION"],
        r"aset|liabilitas|ekuitas": ["ASSET"],
    }
    for pattern, mapped in intent_map.items():
        if re.search(pattern, query, re.IGNORECASE):
            tags.update(mapped)
    return sorted(tags)


# ----------------------------------------------------------
# 2. Reranking
# ----------------------------------------------------------
def tag_overlap_score(chunk_tags: List[str], query_tags: List[str]) -> float:
    """Jaccard-ish: berapa banyak tag query yang tertutup oleh chunk."""
    if not query_tags:
        return 0.0
    return len(set(chunk_tags) & set(query_tags)) / len(set(query_tags))


def rerank(chunks: List[Chunk], query_tags: List[str]) -> List[Chunk]:
    """
    skor_akhir = w_sim * similarity + w_tag * tag_overlap

    Set weight_tag_match=0.0 di config -> tagging mati total, tapi
    pipeline tetap jalan. Itu baseline ablation lo.
    """
    for c in chunks:
        c.score = (
            CONTEXT.weight_similarity * c.score
            + CONTEXT.weight_tag_match * tag_overlap_score(c.tags, query_tags)
        )
    return sorted(chunks, key=lambda c: c.score, reverse=True)


# ----------------------------------------------------------
# 3 & 4. Budgeting + reordering
# ----------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Estimasi cepat. Untuk angka final di tesis, pakai ledger dari API."""
    return int(len(text) / CONTEXT.chars_per_token)


def apply_budget(chunks: List[Chunk]) -> List[Chunk]:
    """Ambil chunk dari skor tertinggi sampai budget token habis."""
    selected: List[Chunk] = []
    used = 0
    for c in chunks:
        if len(selected) >= CONTEXT.max_chunks:
            break
        cost = estimate_tokens(c.text)
        if used + cost > CONTEXT.max_context_tokens:
            continue  # lewati yang kegedean, siapa tahu ada chunk kecil relevan
        selected.append(c)
        used += cost
    return selected


def reorder_against_lost_in_the_middle(chunks: List[Chunk]) -> List[Chunk]:
    """
    Susun ulang: [terbaik, ke-3, ke-5, ..., ke-6, ke-4, ke-2].

    Efeknya chunk paling relevan mendarat di posisi awal DAN akhir
    context — dua posisi yang menurut Liu et al. (2024) paling
    diperhatikan model. Yang lemah dibuang ke tengah.
    """
    if not CONTEXT.reorder_lost_in_the_middle or len(chunks) < 3:
        return chunks
    head, tail = [], []
    for i, c in enumerate(chunks):
        (head if i % 2 == 0 else tail).append(c)
    return head + tail[::-1]


# ----------------------------------------------------------
# Buffer percakapan
# ----------------------------------------------------------
@dataclass
class ConversationBuffer:
    """
    |s.b| dalam formula Cost(s) — dan sumber context bloat paling
    diam-diam. Tanpa pemangkasan, sesi 20 giliran = riwayat 20 giliran
    ikut dikirim tiap kali.

    Buffer ini SENGAJA tidak dibatasi jumlah turn: tujuan eksperimen
    adalah menguji kemampuan context tak terbatas yang dikelola lewat
    KG retrieval (Layer 1) + context engineering (Layer 2), bukan lewat
    pemangkasan riwayat. Pengendalian ukuran tetap ada di apply_budget()
    (max_context_tokens / max_chunks), bukan di sini.
    """
    turns: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        # Tanpa batas turn: uji kemampuan context tak terbatas via KG retrieval
        # + context engineering (bukan pemangkasan riwayat).
        self.turns.append((question, answer))

    def render(self, max_chars_per_turn: int = 300) -> str:
        if not self.turns:
            return ""
        parts = [
            f"[Giliran {i}] T: {q[:150]}\nJ: {a[:max_chars_per_turn]}"
            for i, (q, a) in enumerate(self.turns, 1)
        ]
        return "RIWAYAT SINGKAT:\n" + "\n".join(parts)

    def clear(self) -> None:
        self.turns = []


# ----------------------------------------------------------
# Orkestrasi layer
# ----------------------------------------------------------
@dataclass
class ManagedContext:
    """Output Layer 2 — inilah yang benar-benar dikirim ke model."""
    text: str
    chunks: List[Chunk]
    query_tags: List[str]
    estimated_tokens: int
    dropped_chunks: int


def build_context(
    chunks: List[Chunk],
    query: str,
    buffer: ConversationBuffer | None = None,
    kg_context: str = "",
) -> ManagedContext:
    """
    Pipeline Layer 2 lengkap: tag -> rerank -> budget -> reorder -> render.

    Format render sengaja pakai blok <SUMBER id=...> supaya model punya
    handle eksplisit untuk menyitir. Tanpa ID yang bisa dikutip,
    traceability (H4) cuma jadi klaim, bukan sesuatu yang bisa diukur.
    """
    tagged = tag_chunks(chunks)
    qtags = tag_query(query)
    ranked = rerank(tagged, qtags)
    selected = apply_budget(ranked)
    ordered = reorder_against_lost_in_the_middle(selected)

    blocks = []
    for c in ordered:
        tag_str = ",".join(c.tags) if c.tags else "-"
        blocks.append(
            f"<SUMBER id=\"{c.chunk_id}\" halaman=\"{c.page}\" tag=\"{tag_str}\">\n"
            f"{c.text.strip()}\n"
            f"</SUMBER>"
        )
    body = "\n\n".join(blocks)

    if kg_context:
        body += f"\n\n<GRAF_ENTITAS>\n{kg_context}\n</GRAF_ENTITAS>"
    if buffer and buffer.turns:
        body += f"\n\n{buffer.render()}"

    return ManagedContext(
        text=body,
        chunks=ordered,
        query_tags=qtags,
        estimated_tokens=estimate_tokens(body),
        dropped_chunks=len(chunks) - len(ordered),
    )
