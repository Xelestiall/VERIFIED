"""
runner.py — Eksekutor eksperimen batch.

Menjalankan: 30 dokumen x 3 skenario x 8 pertanyaan -> satu CSV.
Runner harus dikasih limit, biar ga kena RTO
Pakai:
    python runner.py --pdf-dir ./data/pdf --limit 3
    python runner.py --scenarios verified --resume
"""

from __future__ import annotations

import argparse
import csv
import os
import traceback
from typing import Any, Dict, List, Optional, Set

from config import EVAL_QUESTIONS, LOG_FILE, RESULTS_DIR, SCENARIOS
from layer1_retrieval import RetrievalLayer
from llm import LLMClient, TokenLedger
from pipeline import PipelineResult, build_pipeline
from sources import build_source, make_doc_id


# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
CSV_FIELDS = [
    "doc_id", "scenario", "question_id", "question", "answer",
    "confidence", "risk_level", "needs_human", "human_action",
    "numbers_total", "numbers_verified", "numbers_unsupported", "gate_passed",
    "citations", "retrieved_chunks", "query_tags",
    "critic_issues", "revised", "verdict",
    "input_tokens", "output_tokens", "total_tokens", "llm_calls",
    "cost_usd", "latency_s",
    "human_prompts", "trigger_reasons",
    "is_hallucination", "hallucination_type", "annotator_note",
    # Kolom is_hallucination, hallucination_type, annontator_note SENGAJA dikosongkan untuk diisi saat evaluasi
]


class ResultLogger:
    def __init__(self, path: str = LOG_FILE):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def write(self, result: PipelineResult) -> None:
        row = result.to_row()
        row.setdefault("is_hallucination", "")
        row.setdefault("hallucination_type", "")
        row.setdefault("annotator_note", "")
        clean = {k: row.get(k, "") for k in CSV_FIELDS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(clean)

    def completed_keys(self) -> Set[str]:
        """Kunci run yang sudah selesai -> dipakai untuk --resume."""
        if not os.path.exists(self.path):
            return set()
        done: Set[str] = set()
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(f"{row['doc_id']}|{row['scenario']}|{row['question_id']}")
        return done


# ----------------------------------------------------------
# Runner
# ----------------------------------------------------------
class ExperimentRunner:
    def __init__(
        self,
        pdf_paths: List[str],
        scenarios: Optional[List[str]] = None,
        questions: Optional[List[Dict[str, str]]] = None,
        embedding_provider: str = "voyage",
        enable_kg: bool = False,
        logger: Optional[ResultLogger] = None,
    ):
        self.pdf_paths = pdf_paths
        self.scenarios = scenarios or SCENARIOS
        self.questions = questions or EVAL_QUESTIONS
        self.embedding_provider = embedding_provider
        self.enable_kg = enable_kg
        self.logger = logger or ResultLogger()
        self.llm = LLMClient(ledger=TokenLedger())

    def run_document(self, pdf_path: str, resume: bool = False) -> List[PipelineResult]:
        # Pakai make_doc_id() dari sources.py, JANGAN hitung sendiri di sini.
        # Nilai ini jadi kunci join antara run_log.csv dan manifest.csv, dan
        # juga nama file index FAISS. Dua rumus berbeda = manifest tidak bisa
        # dipasangkan dengan hasil run.
        doc_id = make_doc_id(pdf_path)
        print(f"\n{'=' * 60}\nDOKUMEN: {doc_id}\n{'=' * 60}")

        done = self.logger.completed_keys() if resume else set()
        results: List[PipelineResult] = []

        retriever = None
        if any(s != "vanilla" for s in self.scenarios):
            print("  [Layer 1] Membangun index...")
            retriever = RetrievalLayer(pdf_path, self.embedding_provider).build()
            print(f"  [Layer 1] {len(retriever.chunks)} chunk siap.")

        kg = None
        if self.enable_kg and retriever is not None:
            from layer1_retrieval import KnowledgeGraph
            print("  [Layer 1] Ekstraksi knowledge graph...")
            kg = KnowledgeGraph()
            kg.extract_from_chunks(retriever.chunks, self.llm)
            print(f"  [Layer 1] Graf: {kg.stats()}")

        for scenario in self.scenarios:
            kwargs: Dict[str, Any] = {}
            if scenario == "verified" and kg is not None:
                kwargs["kg"] = kg
            pipe = build_pipeline(scenario, retriever=retriever, llm=self.llm, **kwargs)

            for q in self.questions:
                key = f"{doc_id}|{scenario}|{q['id']}"
                if key in done:
                    print(f"  [SKIP] {key}")
                    continue
                try:
                    res = pipe.run(
                        question=q["text"],
                        question_id=q["id"],
                        question_risk=q.get("risk", "medium"),
                        doc_id=doc_id,
                        company_hint=doc_id if scenario == "vanilla" else "",
                    )
                    res.doc_id = doc_id
                    self.logger.write(res)
                    results.append(res)
                    flag = "HUMAN" if res.needs_human else "auto "
                    print(
                        f"  [{scenario:13s}] {q['id']} conf={res.confidence:.2f} "
                        f"tok={res.total_tokens:6d} {flag} "
                        f"unsup={res.numbers_unsupported}"
                    )
                except Exception as e:  # noqa: BLE001
                    # Satu pertanyaan gagal tidak boleh membunuh run 720 respons.
                    print(f"  [ERROR] {key}: {e}")
                    traceback.print_exc(limit=1)
        return results

    def run_all(self, resume: bool = False) -> List[PipelineResult]:
        all_results: List[PipelineResult] = []
        for path in self.pdf_paths:
            all_results.extend(self.run_document(path, resume=resume))
        self.summary(all_results)
        return all_results

    @staticmethod
    def summary(results: List[PipelineResult]) -> None:
        """Ringkasan cepat di terminal. Analisis serius ada di evaluation.py."""
        if not results:
            print("\nTidak ada hasil.")
            return
        print(f"\n{'=' * 60}\nRINGKASAN ({len(results)} respons)\n{'=' * 60}")
        print(f"{'skenario':<15}{'n':>5}{'avg_tok':>10}{'avg_conf':>10}"
              f"{'unsup':>8}{'human%':>9}")
        for s in sorted({r.scenario for r in results}):
            rs = [r for r in results if r.scenario == s]
            n = len(rs)
            print(
                f"{s:<15}{n:>5}"
                f"{sum(r.total_tokens for r in rs) / n:>10.0f}"
                f"{sum(r.confidence for r in rs) / n:>10.2f}"
                f"{sum(r.numbers_unsupported for r in rs):>8}"
                f"{sum(r.needs_human for r in rs) / n * 100:>8.0f}%"
            )
        total_cost = sum(r.cost_usd for r in results)
        print(f"\nTotal biaya API: ${total_cost:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Runner eksperimen VERIFIED")
    p.add_argument("--source", default="local",
                   choices=["local", "supabase", "gdrive", "sharepoint"],
                   help="asal dokumen; selain 'local' akan diunduh dulu ke --pdf-dir")
    p.add_argument("--pdf-dir", default="./data/pdf")
    p.add_argument("--pdf", nargs="*", help="path PDF spesifik")
    # Dipakai build_source() untuk source non-local. Lihat sources.py.
    p.add_argument("--bucket", default="", help="[supabase] nama bucket")
    p.add_argument("--prefix", default="", help="[supabase/sharepoint] subfolder")
    p.add_argument("--folder-id", default="", help="[gdrive] ID folder Drive")
    p.add_argument("--site-id", default="", help="[sharepoint] site ID Graph")
    p.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    p.add_argument("--questions", nargs="*", help="filter ID pertanyaan, mis. Q1 Q2")
    p.add_argument("--limit", type=int, help="batasi jumlah dokumen (buat pilot)")
    p.add_argument("--embedding", default="voyage", choices=["voyage", "local"])
    p.add_argument("--enable-kg", action="store_true")
    p.add_argument("--resume", action="store_true", help="lewati yang sudah ada di CSV")
    p.add_argument("--out", default=LOG_FILE)
    args = p.parse_args()

    # --pdf menang atas segalanya. Selain itu serahkan ke sources.py: untuk
    # 'local' dia cuma nge-glob folder, untuk sumber cloud dia mengunduh ke
    # cache lalu menulis results/manifest.csv (sha256 tiap dokumen) — lampiran
    # bukti reproducibility yang dipasangkan dengan run_log.csv lewat doc_id.
    if args.pdf:
        paths = sorted(args.pdf)
    else:
        try:
            paths = build_source(args).sync(cache_dir=args.pdf_dir)
        except RuntimeError as e:
            raise SystemExit(str(e))

    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"Tidak ada PDF di {args.pdf_dir}")

    questions = EVAL_QUESTIONS
    if args.questions:
        questions = [q for q in EVAL_QUESTIONS if q["id"] in args.questions]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Dokumen : {len(paths)}")
    print(f"Skenario: {args.scenarios}")
    print(f"Soal    : {[q['id'] for q in questions]}")
    print(f"Estimasi: {len(paths) * len(args.scenarios) * len(questions)} respons")

    ExperimentRunner(
        pdf_paths=paths, scenarios=args.scenarios, questions=questions,
        embedding_provider=args.embedding, enable_kg=args.enable_kg,
        logger=ResultLogger(args.out),
    ).run_all(resume=args.resume)


if __name__ == "__main__":
    main()
