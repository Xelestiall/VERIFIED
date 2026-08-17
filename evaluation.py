"""
evaluation.py — Perhitungan metrik sesuai Tabel 3.1 tesis.

  Retrieval Accuracy -> Context Precision & Recall (RAGAS)
  Context Efficiency -> Token count per respons/sesi
  Hallucination Rate -> HR = (respons mengandung kesalahan faktual / total) x 100%

Hallucination rate: ini metrik yang butuh menghitung `is_hallucination` yang diisi evaluator secara manual.
"""

from __future__ import annotations

import argparse
import csv
import os
import pandas as pd
from typing import Any, Dict, List, Optional, Set

# ----------------------------------------------------------
# Logging untuk evaluasi model Context Engineering compared with standard RAG dan Vanila LLM
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["confidence", "total_tokens", "input_tokens", "output_tokens",
                "numbers_total", "numbers_verified", "numbers_unsupported",
                "cost_usd", "latency_s", "llm_calls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["needs_human", "gate_passed", "revised"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df

# ----------------------------------------------------------
# Hallucination rate
def hallucination_rate(df: pd.DataFrame, annotated_only: bool = True) -> pd.DataFrame:
    #is_hallucination -> TRUE/FALSE.
    d = df.copy()
    d["is_hallucination"] = pd.to_numeric(d["is_hallucination"], errors="coerce")
    if annotated_only:
        d = d[d["is_hallucination"].notna()]
    if d.empty:
        raise ValueError("Belum ada baris yang dianotasi.")

    out = d.groupby("scenario").agg(
        n_annotated=("is_hallucination", "size"),
        n_hallucinated=("is_hallucination", "sum"),
    ).reset_index()
    out["hallucination_rate_pct"] = (
        out["n_hallucinated"] / out["n_annotated"] * 100
    ).round(2)
    return out

def hallucination_by_type(df: pd.DataFrame) -> pd.DataFrame:
    #hallucination_type -> factual_fabrication, factual_contradiction, instruction, extrinsic.
    d = df[df["hallucination_type"].notna() & (df["hallucination_type"] != "")]
    if d.empty:
        return pd.DataFrame()
    return pd.crosstab(d["hallucination_type"], d["scenario"], margins=True)


def auto_proxy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy otomatis: berapa persen respons memuat angka tak terdukung.

    Gunanya: (a) prioritaskan baris mana yang dianotasi manusia duluan,
    (b) bandingkan dengan HR manual untuk mengukur seberapa baik gate
    numerik menangkap halusinasi nyata. Angka kedua ini menarik
    sebagai temuan tersendiri.
    """
    d = df.copy()
    d["has_unsupported"] = d["numbers_unsupported"] > 0
    out = d.groupby("scenario").agg(
        n=("has_unsupported", "size"),
        n_with_unsupported=("has_unsupported", "sum"),
        total_numbers=("numbers_total", "sum"),
        total_unsupported=("numbers_unsupported", "sum"),
    ).reset_index()
    out["unsupported_response_pct"] = (
        out["n_with_unsupported"] / out["n"] * 100
    ).round(2)
    out["unsupported_number_pct"] = (
        out["total_unsupported"] / out["total_numbers"].replace(0, pd.NA) * 100
    ).round(2)
    return out


# ----------------------------------------------------------
#   Token Cost
def token_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("scenario").agg(
        n=("total_tokens", "size"),
        avg_input=("input_tokens", "mean"),
        avg_output=("output_tokens", "mean"),
        avg_total=("total_tokens", "mean"),
        median_total=("total_tokens", "median"),
        avg_calls=("llm_calls", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    ).round(2).reset_index()
    return out

def token_growth_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pertumbuhan token sepanjang sesi (Q1 -> Q8).
    kurvanya lebih landai, karena Layer 2 mencegah buffer menumpuk
    """
    d = df.copy()
    d["turn"] = d["question_id"].str.extract(r"(\d+)").astype(float)
    return (
        d.groupby(["scenario", "turn"])["total_tokens"]
        .mean().round(0).unstack(0)
    )


def efficiency_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rasio efisiensi: berapa token dibayar per satu angka terverifikasi.

    Metrik gabungan akurasi-biaya. Berguna kalau VERIFIED kalah di
    token mentah tapi menang telak di kualitas — dan itu skenario yang
    sangat mungkin terjadi.
    """
    out = df.groupby("scenario").agg(
        total_tokens=("total_tokens", "sum"),
        verified_numbers=("numbers_verified", "sum"),
    ).reset_index()
    out["tokens_per_verified_number"] = (
        out["total_tokens"] / out["verified_numbers"].replace(0, pd.NA)
    ).round(1)
    return out


# ----------------------------------------------------------
# H4 — Traceability
# ----------------------------------------------------------
def traceability_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Persentase respons yang membawa sitasi [chunk_id] yang bisa dilacak."""
    d = df.copy()
    d["n_citations"] = (
        d["citations"].fillna("").astype(str)
        .apply(lambda s: len([x for x in s.split(";") if x.strip()]))
    )
    d["has_citation"] = d["n_citations"] > 0
    out = d.groupby("scenario").agg(
        n=("has_citation", "size"),
        n_with_citation=("has_citation", "sum"),
        avg_citations=("n_citations", "mean"),
    ).reset_index()
    out["traceable_pct"] = (out["n_with_citation"] / out["n"] * 100).round(2)
    out["avg_citations"] = out["avg_citations"].round(2)
    return out


# ----------------------------------------------------------
# Layer 4 — Sociotechnical
# ----------------------------------------------------------
def human_trigger_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seberapa sering Layer 4 memanggil manusia, dipecah per level risiko.

    Angka ini yang menjawab pertanyaan penguji "apakah ini malah
    menambah beban analis?". Kalau trigger nyala di >60% respons,
    naikkan ambang di config — desainnya belum selektif.
    """
    d = df[df["scenario"] == "verified"]
    if d.empty:
        return pd.DataFrame()
    out = d.groupby("risk_level").agg(
        n=("needs_human", "size"),
        n_escalated=("needs_human", "sum"),
        avg_confidence=("confidence", "mean"),
    ).reset_index()
    out["escalation_pct"] = (out["n_escalated"] / out["n"] * 100).round(2)
    out["avg_confidence"] = out["avg_confidence"].round(3)
    return out


# ----------------------------------------------------------
# External judge confidence (Opsi B)
# ----------------------------------------------------------
# PipelineResult.confidence di run_log.csv SENGAJA 0.0 untuk Standard RAG
# & Vanilla LLM (lihat pipeline.py) -- itu bukan bug, itu bukti bahwa
# baseline tidak punya mekanisme self-assessment. Salah satu novelty claim
# VERIFIED (lihat docstring app.py: "Kalau sistem ragu, dia BILANG ragu")
# justru bergantung pada baseline TIDAK bisa melakukan itu.
#
# Fungsi di bawah TIDAK mengubah confidence bawaan itu. Ia menjalankan
# SCORER_SYSTEM yang identik (dari layer3_verification.py) SEKALI secara
# post-hoc ke jawaban ketiga skenario, sebagai "external judge" yang
# terpisah dari pipeline produksi -- supaya confidence bisa dibandingkan
# apples-to-apples di bab evaluasi tanpa merusak temuan aslinya:
#   - verified      -> dinilai terhadap context Layer 2 (sudah terkurasi)
#   - standard_rag  -> dinilai terhadap raw top-k chunk (retrieved_chunks)
#   - vanilla       -> dinilai TANPA sumber sama sekali -> wajar kalau
#                      judged_confidence-nya rendah, itu memang temuan H1
JUDGE_CSV_FIELDS = [
    "doc_id", "scenario", "question_id", "judged_confidence", "judged_reasoning",
]


class JudgeLogger:
    """Pola sama dengan ResultLogger (runner.py): append per-baris + resume,
    supaya penilaian 720 respons tidak hilang kalau kepotong di tengah."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=JUDGE_CSV_FIELDS).writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=JUDGE_CSV_FIELDS).writerow(
                {k: row.get(k, "") for k in JUDGE_CSV_FIELDS}
            )

    def completed_keys(self) -> Set[str]:
        if not os.path.exists(self.path):
            return set()
        done: Set[str] = set()
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(f"{row['doc_id']}|{row['scenario']}|{row['question_id']}")
        return done


def load_chunk_lookup(doc_id: str) -> Dict[str, str]:
    """Baca teks chunk dari cache pickle Layer 1 (RETRIEVAL.index_dir).

    Sengaja TIDAK memanggil retriever.build() -- itu akan re-embed via
    Voyage API. Index/pickle-nya sudah ada di disk dari run pertama
    (app.py / runner.py), jadi cukup di-load langsung, gratis.
    """
    import pickle
    from config import RETRIEVAL

    meta_path = os.path.join(RETRIEVAL.index_dir, f"{doc_id}_chunks.pkl")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return {c.chunk_id: c.text for c in chunks}


def _context_for_row(row: "pd.Series", cache: Dict[str, Dict[str, str]]) -> str:
    """Rakit ulang blok <SUMBER> persis seperti yang dilihat sistem asal,
    dari kolom `retrieved_chunks` (id) + chunk text dari disk."""
    doc_id = str(row.get("doc_id", ""))
    if row.get("scenario") == "vanilla" or not doc_id:
        return ""  # Vanilla memang tidak pernah diberi sumber apa pun.
    ids = [c.strip() for c in str(row.get("retrieved_chunks", "")).split(";") if c.strip()]
    if not ids:
        return ""
    lookup = cache.setdefault(doc_id, load_chunk_lookup(doc_id))
    return "\n\n".join(
        f'<SUMBER id="{cid}">\n{lookup[cid]}\n</SUMBER>'
        for cid in ids if cid in lookup
    )


def run_judge_confidence(
    csv_path: str,
    out_path: str = "./results/judged_confidence.csv",
    resume: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Jalankan Opsi B untuk SEMUA baris di run_log.csv (ketiga skenario),
    tulis ke CSV TERPISAH (default results/judged_confidence.csv) lewat
    JudgeLogger, lalu kembalikan hasil join-nya. run_log.csv (arsip utama,
    dibaca ulang oleh runner.py --resume) TIDAK disentuh sama sekali.

    Pakai:
        python evaluation.py --csv results/run_log.csv --judge-confidence
        python evaluation.py --judge-confidence --judge-limit 20   # pilot
    """
    from llm import LLMClient
    from layer3_verification import SCORER_SYSTEM

    df = load_results(csv_path)
    logger = JudgeLogger(out_path)
    done = logger.completed_keys() if resume else set()

    llm = LLMClient()
    chunk_cache: Dict[str, Dict[str, str]] = {}
    n_judged = 0

    for _, row in df.iterrows():
        key = f"{row['doc_id']}|{row['scenario']}|{row['question_id']}"
        if key in done:
            continue
        if limit and n_judged >= limit:
            break

        context = _context_for_row(row, chunk_cache)
        scored = llm.complete_json(
            f"PERTANYAAN:\n{row['question']}\n\n"
            f"SUMBER:\n{context or '(tidak ada sumber diberikan ke sistem ini)'}\n\n"
            f"JAWABAN YANG DINILAI:\n{row['answer']}\n\n"
            'Nilai jawaban. Format: {"confidence":0.0-1.0,'
            '"supported_claims":["..."],"unsupported_claims":["..."],'
            '"reasoning":"ringkas, maksimal 2 kalimat"}',
            system=SCORER_SYSTEM,
            stage="judge_post_hoc",
            fallback={"confidence": 0.0, "reasoning": "parse_error"},
        )
        logger.write({
            "doc_id": row["doc_id"], "scenario": row["scenario"],
            "question_id": row["question_id"],
            "judged_confidence": round(float(scored.get("confidence", 0.0)), 3),
            "judged_reasoning": scored.get("reasoning", ""),
        })
        n_judged += 1
        print(f"  [judge] {key} -> {float(scored.get('confidence', 0.0)):.2f}")

    print(f"\nJudge selesai: {n_judged} baris baru dinilai. "
          f"Biaya sesi ini: ${llm.ledger.cost_usd():.4f}")

    judged_df = pd.read_csv(out_path)
    return df.merge(
        judged_df[["doc_id", "scenario", "question_id",
                    "judged_confidence", "judged_reasoning"]],
        on=["doc_id", "scenario", "question_id"], how="left",
    )


def confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bandingkan confidence bawaan sistem (`confidence`, cuma bermakna utk
    VERIFIED) dengan `judged_confidence` (seragam 3 skenario, dari
    run_judge_confidence). Perlu di-merge dulu -- lihat run_judge_confidence().
    """
    if "judged_confidence" not in df.columns:
        raise ValueError(
            "Kolom judged_confidence belum ada -- jalankan run_judge_confidence() dulu."
        )
    out = df.groupby("scenario").agg(
        n=("judged_confidence", "size"),
        avg_self_confidence=("confidence", "mean"),
        avg_judged_confidence=("judged_confidence", "mean"),
    ).round(3).reset_index()
    out["note"] = out["scenario"].map({
        "verified": "self-assessment asli (Scorer/Critic/Commander)",
        "standard_rag": "confidence=0.0 by design; judged = external judge",
        "vanilla": "confidence=0.0 by design; judged = external judge tanpa sumber",
    })
    return out


# ----------------------------------------------------------
# RAGAS (opsional)
# ----------------------------------------------------------
def build_ragas_dataset(
    df: pd.DataFrame,
    chunk_lookup: Dict[str, str],
    ground_truth: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Siapkan payload RAGAS.

    chunk_lookup : {chunk_id: teks} — ambil dari retriever.chunks
    ground_truth : {(doc_id, question_id): jawaban acuan} yang lo tulis
                   manual dari laporan asli. Tanpa ini, context_recall
                   tidak bisa dihitung — dan itu setengah dari dimensi
                   Retrieval Accuracy lo, jadi sediakan waktu untuknya.
    """
    rows = []
    for _, r in df.iterrows():
        ids = [c.strip() for c in str(r.get("retrieved_chunks", "")).split(";") if c.strip()]
        contexts = [chunk_lookup[i] for i in ids if i in chunk_lookup]
        item = {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": contexts,
        }
        if ground_truth:
            key = f"{r['doc_id']}|{r['question_id']}"
            if key in ground_truth:
                item["ground_truth"] = ground_truth[key]
        rows.append(item)
    return rows


def run_ragas(dataset_rows: List[Dict[str, Any]]) -> Any:
    """
    Hitung context precision & recall.

    RAGAS memakai LLM sebagai judge, jadi ini menimbulkan biaya API
    tambahan DAN pertanyaan metodologis: model menilai output model.
    Mitigasinya: pakai model judge yang BERBEDA dari generator, dan
    validasi silang sebagian sampel secara manual. Tulis ini di
    limitasi tesis.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    ds = Dataset.from_list(dataset_rows)
    metrics = [context_precision, faithfulness]
    if any("ground_truth" in r for r in dataset_rows):
        metrics.append(context_recall)
    return evaluate(ds, metrics=metrics)


# ----------------------------------------------------------
# Laporan
# ----------------------------------------------------------
def full_report(csv_path: str, out_dir: str = "./results") -> Dict[str, pd.DataFrame]:
    df = load_results(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    report: Dict[str, pd.DataFrame] = {
        "token_analysis": token_analysis(df),
        "token_growth": token_growth_by_turn(df),
        "efficiency_ratio": efficiency_ratio(df),
        "traceability": traceability_analysis(df),
        "auto_proxy": auto_proxy_metrics(df),
        "human_trigger": human_trigger_analysis(df),
    }
    try:
        report["hallucination_rate"] = hallucination_rate(df)
        report["hallucination_by_type"] = hallucination_by_type(df)
    except ValueError as e:
        print(f"[info] HR dilewati: {e}")

    for name, table in report.items():
        if table is None or table.empty:
            continue
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        print(table.to_string())
        table.to_csv(os.path.join(out_dir, f"metric_{name}.csv"))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluasi hasil eksperimen VERIFIED")
    p.add_argument("--csv", default="./results/run_log.csv")
    p.add_argument("--out", default="./results")
    p.add_argument("--ragas", action="store_true", help="jalankan RAGAS (butuh biaya API)")
    p.add_argument(
        "--judge-confidence", action="store_true",
        help="Opsi B: nilai SEMUA jawaban (3 skenario) pakai Scorer prompt "
             "yang sama, post-hoc (butuh biaya API, 1 call/baris). Tidak "
             "mengubah confidence bawaan di run_log.csv.",
    )
    p.add_argument(
        "--judge-limit", type=int, default=None,
        help="batasi jumlah baris baru yang dinilai per run --judge-confidence (buat pilot)",
    )
    p.add_argument("--judge-out", default="./results/judged_confidence.csv")
    args = p.parse_args()

    full_report(args.csv, args.out)
    if args.ragas:
        print("\n[RAGAS] Butuh chunk_lookup + ground_truth. "
              "Lihat build_ragas_dataset() dan panggil dari notebook.")

    if args.judge_confidence:
        print(f"\n{'=' * 60}\nJUDGE CONFIDENCE (Opsi B)\n{'=' * 60}")
        judged = run_judge_confidence(
            args.csv, out_path=args.judge_out, limit=args.judge_limit,
        )
        calib = confidence_calibration(judged)
        print(calib.to_string())
        calib.to_csv(os.path.join(args.out, "metric_confidence_calibration.csv"), index=False)


if __name__ == "__main__":
    main()
