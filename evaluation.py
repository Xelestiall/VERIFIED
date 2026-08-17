"""
evaluation.py — Perhitungan metrik sesuai Tabel 3.1 tesis.

  Retrieval Accuracy -> Context Precision & Recall (RAGAS)
  Context Efficiency -> Token count per respons/sesi
  Hallucination Rate -> HR = (respons mengandung kesalahan faktual / total) x 100%

CATATAN PENTING soal hallucination rate: ini metrik yang butuh
ANOTASI MANUSIA. Skrip di bawah menghitungnya dari kolom
`is_hallucination` yang lo isi manual di CSV.

python evaluation.py --csv results/run_log.csv
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
from typing import Any, Dict, List, Optional

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
# Layer 4 — beban kerja manusia
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
    args = p.parse_args()

    full_report(args.csv, args.out)
    if args.ragas:
        print("\n[RAGAS] Butuh chunk_lookup + ground_truth. "
              "Lihat build_ragas_dataset() dan panggil dari notebook.")


if __name__ == "__main__":
    main()
