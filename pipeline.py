"""
pipeline.py — Perakitan seluruh layer + 3 skenario eksperimen.

Tiga skenario tesis, dibedakan HANYA oleh layer yang aktif:

  vanilla       -> tanpa retrieval. Murni based ondata  training model
  standard_rag  -> Layer 1 saja. Top-k mentah, langsung ke model.
  verified      -> Layer 1 + 2 + 3 + 4.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import CONTEXT, MODEL
from layer1_retrieval import Chunk, KnowledgeGraph, RetrievalLayer
from layer2_context import ConversationBuffer, build_context, tag_chunks, tag_query
from layer3_verification import numerical_gate, run_debate
from human_review import human_judgement_trigger
from llm import LLMClient, TokenLedger


# ==========================================================
# Prompt
# ==========================================================
BASE_SYSTEM = """Anda adalah asisten analisis laporan keuangan tahunan perusahaan
terbuka Indonesia. Anda menjawab dalam Bahasa Indonesia, ringkas dan presisi."""

GROUNDED_SYSTEM = BASE_SYSTEM + """

ATURAN WAJIB:
1. Jawab HANYA berdasarkan blok <SUMBER> yang diberikan.
2. Setiap angka dan klaim faktual WAJIB diikuti sitasi [chunk_id] sesuai
   atribut id pada blok sumbernya.
3. Jika informasi tidak ada di sumber, tulis persis:
   "Tidak ditemukan dalam dokumen yang tersedia."
   JANGAN menebak, JANGAN melengkapi dari pengetahuan umum.
4. Jangan mengubah, membulatkan, atau menghitung ulang angka yang ada
   di sumber kecuali diminta secara eksplisit."""


# ==========================================================
# Hasil
# ==========================================================
@dataclass
class PipelineResult:
    scenario: str
    question_id: str
    question: str
    answer: str
    doc_id: str = ""
    confidence: float = 1.0
    citations: List[str] = field(default_factory=list)
    retrieved_chunks: List[str] = field(default_factory=list)
    query_tags: List[str] = field(default_factory=list)

    # Layer 3
    numbers_total: int = 0
    numbers_verified: int = 0
    numbers_unsupported: int = 0
    gate_passed: bool = True
    critic_issues: List[str] = field(default_factory=list)
    revised: bool = False
    verdict: str = "kept"

    # Layer 4
    risk_level: str = "medium"
    needs_human: bool = False
    human_action: str = "auto_accept"
    human_prompts: List[str] = field(default_factory=list)
    trigger_reasons: List[str] = field(default_factory=list)

    # Biaya
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        """Ratakan jadi satu baris CSV (list -> string dipisah ';')."""
        row = asdict(self)
        for k, v in row.items():
            if isinstance(v, list):
                row[k] = "; ".join(str(x) for x in v)
        return row


# ==========================================================
# Skenario 1 — Vanilla LLM
# ==========================================================
class VanillaPipeline:
    """Baseline terlemah karena tanpa dokumen sama sekali.
    Membuktikan bahwa jawaban lain memang datang dari retrieval, bukan dari model yang
    kebetulan hafal data latihannya
    """
    scenario = "vanilla"

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def run(self, question: str, question_id: str = "", doc_id: str = "",
            company_hint: str = "", **_: Any) -> PipelineResult:
        self.llm.ledger.reset()
        t0 = time.time()

        prompt = question
        if company_hint:
            prompt = f"Untuk perusahaan {company_hint}: {question}"

        answer = self.llm.complete(prompt, system=BASE_SYSTEM, stage="generation")
        snap = self.llm.ledger.snapshot()

        return PipelineResult(
            scenario=self.scenario, question_id=question_id, question=question,
            answer=answer, doc_id=doc_id, confidence=0.0,
            input_tokens=snap["input_tokens"], output_tokens=snap["output_tokens"],
            total_tokens=snap["total_tokens"], llm_calls=snap["llm_calls"],
            cost_usd=snap["cost_usd"], latency_s=round(time.time() - t0, 2),
        )


# ==========================================================
# Skenario 2 — Standard RAG
class StandardRAGPipeline:
    """
    Baseline yang biasa digunakan user. Tetapi tidak ada tagging, tidak ada budget
    kontekstual, tidak ada critique, tidak ada trigger manusia.
    """
    scenario = "standard_rag"

    def __init__(self, retriever: RetrievalLayer, llm: Optional[LLMClient] = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def run(self, question: str, question_id: str = "", **_: Any) -> PipelineResult:
        self.llm.ledger.reset()
        t0 = time.time()

        chunks = self.retriever.search(question)
        context = "\n\n".join(
            f'<SUMBER id="{c.chunk_id}" halaman="{c.page}">\n{c.text}\n</SUMBER>'
            for c in chunks
        )
        answer = self.llm.complete(
            f"SUMBER:\n{context}\n\nPERTANYAAN:\n{question}",
            system=GROUNDED_SYSTEM, stage="generation",
        )

        # Gate numerik tetap DIHITUNG di sini, tapi tidak dipakai untuk
        # mengoreksi jawaban. Ini murni instrumen pengukuran, supaya
        # hallucination rate ketiga skenario diukur dengan alat yang sama.
        gate = numerical_gate(answer, chunks)
        snap = self.llm.ledger.snapshot()

        return PipelineResult(
            scenario=self.scenario, question_id=question_id, question=question,
            answer=answer, doc_id=self.retriever.doc_id, confidence=0.0,
            citations=extract_citations(answer),
            retrieved_chunks=[c.chunk_id for c in chunks],
            numbers_total=gate.total_numbers, numbers_verified=gate.verified_numbers,
            numbers_unsupported=len(gate.unsupported), gate_passed=gate.passed,
            input_tokens=snap["input_tokens"], output_tokens=snap["output_tokens"],
            total_tokens=snap["total_tokens"], llm_calls=snap["llm_calls"],
            cost_usd=snap["cost_usd"], latency_s=round(time.time() - t0, 2),
        )


# ==========================================================
# Skenario 3 — VERIFIED
class VerifiedPipeline:
    """
    Pipeline lengkap framework yang diusulkan.
    """

    scenario = "verified"

    def __init__(
        self,
        retriever: RetrievalLayer,
        llm: Optional[LLMClient] = None,
        kg: Optional[KnowledgeGraph] = None,
        buffer: Optional[ConversationBuffer] = None,
    ):
        self.retriever = retriever
        self.llm = llm or LLMClient()
        self.kg = kg
        self.buffer = buffer or ConversationBuffer()

    def run(
        self,
        question: str,
        question_id: str = "",
        question_risk: str = "medium",
        risk_override: Optional[str] = None,
        use_buffer: bool = True,
        on_step: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
        **_: Any,
    ) -> PipelineResult:
        self.llm.ledger.reset()
        t0 = time.time()

        # --- LAYER 1 ---
        from config import RETRIEVAL
        candidates = self.retriever.search(question, top_k=RETRIEVAL.top_k_prefetch)

        kg_ctx = ""
        if self.kg is not None:
            qtags_probe = tag_query(question)
            if "SEGMENT" in qtags_probe:
                kg_ctx = self.kg.neighbors_text(self.retriever.doc_id, hops=2)

        # --- LAYER 2 ---
        managed = build_context(
            candidates, question,
            buffer=self.buffer if use_buffer else None,
            kg_context=kg_ctx,
        )

        # --- GENERASI ---
        draft = self.llm.complete(
            f"SUMBER:\n{managed.text}\n\nPERTANYAAN:\n{question}",
            system=GROUNDED_SYSTEM, stage="generation",
        )

        # --- LAYER 3 ---
        gate = numerical_gate(draft, managed.chunks)
        debate = run_debate(self.llm, question, draft, managed.text, gate, on_step=on_step)

        # Gate dijalankan ULANG pada jawaban final. Kalau tidak, jawaban
        # hasil revisi Commander bisa menyelipkan angka baru yang belum
        # pernah diperiksa siapa pun.
        final_gate = numerical_gate(debate.final_answer, managed.chunks)

        # --- LAYER 4 ---
        trigger = human_judgement_trigger(
            question=question, question_risk=question_risk,
            query_tags=managed.query_tags, confidence=debate.confidence,
            gate=final_gate, critic_issues=debate.critic_issues,
            risk_override=risk_override,
        )

        if use_buffer:
            self.buffer.add(question, debate.final_answer)

        snap = self.llm.ledger.snapshot()
        return PipelineResult(
            scenario=self.scenario, question_id=question_id, question=question,
            answer=debate.final_answer, doc_id=self.retriever.doc_id,
            confidence=debate.confidence,
            citations=extract_citations(debate.final_answer),
            retrieved_chunks=[c.chunk_id for c in managed.chunks],
            query_tags=managed.query_tags,
            numbers_total=final_gate.total_numbers,
            numbers_verified=final_gate.verified_numbers,
            numbers_unsupported=len(final_gate.unsupported),
            gate_passed=final_gate.passed,
            critic_issues=debate.critic_issues, revised=debate.revised,
            verdict=debate.verdict,
            risk_level=trigger.risk_level, needs_human=trigger.needs_human,
            human_action=trigger.action, human_prompts=trigger.prompts_for_human,
            trigger_reasons=trigger.reasons,
            input_tokens=snap["input_tokens"], output_tokens=snap["output_tokens"],
            total_tokens=snap["total_tokens"], llm_calls=snap["llm_calls"],
            cost_usd=snap["cost_usd"], latency_s=round(time.time() - t0, 2),
        )


# ==========================================================
# Util
# ==========================================================
def extract_citations(answer: str) -> List[str]:
    """
    Tarik semua [chunk_id] dari jawaban -> Jawaban menggunakan metrik traceability, dan
    lebih meyakinkan daripada klaim kualitatif "output bisa ditelusuri".
    """
    import re
    return sorted(set(
        f"{m.group(1)}::c{m.group(2)}"
        for m in re.finditer(r"\[([^\[\]]+?)::c(\d+)\]", answer)
    ))


def build_pipeline(scenario: str, retriever: Optional[RetrievalLayer] = None,
                   llm: Optional[LLMClient] = None, **kwargs: Any):
    """Factory: string skenario -> objek pipeline."""
    llm = llm or LLMClient(ledger=TokenLedger())
    if scenario == "vanilla":
        return VanillaPipeline(llm=llm)
    if scenario == "standard_rag":
        if retriever is None:
            raise ValueError("standard_rag butuh retriever.")
        return StandardRAGPipeline(retriever=retriever, llm=llm)
    if scenario == "verified":
        if retriever is None:
            raise ValueError("verified butuh retriever.")
        return VerifiedPipeline(retriever=retriever, llm=llm, **kwargs)
    raise ValueError(f"Skenario tidak dikenal: {scenario}")
