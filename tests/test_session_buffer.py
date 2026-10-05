"""
Uji buffer sesi pada baseline (vanilla & standard_rag).

Tanpa memanggil API: LLM diganti FakeLLM yang merekam prompt.
Yang dijaga:
  * use_buffer=False (default) -> prompt Q2 TIDAK memuat riwayat Q1 (soal independen)
  * use_buffer=True            -> prompt Q2 memuat riwayat Q1 lewat ConversationBuffer.render()
  * riwayat baseline identik formatnya dengan VERIFIED (render() yang sama)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_retrieval import Chunk
from llm import TokenLedger
from pipeline import StandardRAGPipeline, VanillaPipeline


class FakeLLM:
    def __init__(self):
        self.ledger = TokenLedger()
        self.prompts = []

    def complete(self, prompt, system="", stage="generation", **_):
        self.prompts.append(prompt)
        return f"jawaban-{len(self.prompts)}"


class FakeRetriever:
    doc_id = "doc"

    def search(self, question, top_k=None):
        return [Chunk("doc::c0001", "isi chunk", page=1)]

    def table_scales(self):
        return set()


def _two_turns(pipe, use_buffer):
    pipe.run(question="pertanyaan satu", question_id="Q1", use_buffer=use_buffer)
    pipe.run(question="pertanyaan dua", question_id="Q2", use_buffer=use_buffer)
    return pipe.llm.prompts


def test_vanilla_independen_secara_default():
    prompts = _two_turns(VanillaPipeline(llm=FakeLLM()), use_buffer=False)
    assert "RIWAYAT" not in prompts[1]
    assert "jawaban-1" not in prompts[1]


def test_vanilla_membawa_riwayat_bila_buffer_aktif():
    prompts = _two_turns(VanillaPipeline(llm=FakeLLM()), use_buffer=True)
    assert "RIWAYAT SINGKAT" not in prompts[0]          # Q1 belum punya riwayat
    assert "pertanyaan satu" in prompts[1] and "jawaban-1" in prompts[1]
    assert prompts[1].rstrip().endswith("pertanyaan dua")


def test_rag_independen_secara_default():
    prompts = _two_turns(StandardRAGPipeline(FakeRetriever(), llm=FakeLLM()), use_buffer=False)
    assert "RIWAYAT" not in prompts[1]


def test_rag_membawa_riwayat_di_antara_sumber_dan_pertanyaan():
    prompts = _two_turns(StandardRAGPipeline(FakeRetriever(), llm=FakeLLM()), use_buffer=True)
    p = prompts[1]
    assert "RIWAYAT SINGKAT" in p and "jawaban-1" in p
    # urutan sama dengan VERIFIED: SUMBER -> riwayat -> PERTANYAAN
    assert p.index("</SUMBER>") < p.index("RIWAYAT SINGKAT") < p.index("PERTANYAAN:")


def test_buffer_tiap_pipeline_terpisah():
    a, b = VanillaPipeline(llm=FakeLLM()), VanillaPipeline(llm=FakeLLM())
    a.run(question="x", use_buffer=True)
    assert len(a.buffer.turns) == 1 and len(b.buffer.turns) == 0
