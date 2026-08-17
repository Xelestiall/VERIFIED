"""
layer4_sociotechnical.py — LAYER 4: SOCIOTECHNICAL VALIDATION

Layer yang bikin VERIFIED bukan sekadar tumpukan trik teknis.

Logika intinya satu kalimat: beban verifikasi manusia diselaraskan
dengan tingkat risiko tugas (Kudina & van de Poel, 2024). Pertanyaan
deskriptif berisiko rendah lolos otomatis; klaim angka berisiko tinggi
harus melewati ambang confidence yang jauh lebih ketat.

Kontras dengan "human-in-the-loop" biasa yang menyuruh manusia memeriksa
SEMUA output — pola itu ditinggalkan analis dalam hitungan hari karena
melelahkan. Di sini triggernya selektif, dan itu justifikasi desainnya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import SOCIOTECHNICAL
from layer3_verification import NumericalGateResult


@dataclass
class HumanTriggerResult:
    needs_human: bool
    risk_level: str
    confidence: float
    threshold: float
    reasons: List[str] = field(default_factory=list)
    prompts_for_human: List[str] = field(default_factory=list)
    action: str = "auto_accept"  # auto_accept | review | escalate


def assess_risk(
    question_risk: str,
    query_tags: List[str],
    override: Optional[str] = None,
) -> str:
    """
    Tentukan risk level final.

    Risiko awal datang dari config (dilabeli manual per pertanyaan),
    lalu bisa DINAIKKAN oleh tag yang terdeteksi Layer 2. Cuma naik,
    nggak pernah turun — kalau pertanyaan deskriptif ternyata menyeret
    angka mata uang, risikonya memang naik.
    """
    if override:
        return override

    order = {"low": 0, "medium": 1, "high": 2}
    level = question_risk if question_risk in order else "medium"

    if any(t in SOCIOTECHNICAL.high_risk_tags for t in query_tags):
        level = "high"
    elif any(t in SOCIOTECHNICAL.medium_risk_tags for t in query_tags):
        if order[level] < order["medium"]:
            level = "medium"
    return level


def build_human_prompts(
    question: str,
    gate: NumericalGateResult,
    critic_issues: List[str],
    risk_level: str,
) -> List[str]:
    """
    Susun pertanyaan konkret untuk analis.

    Ini bagian yang sering dilewatkan orang: jangan cuma bilang
    "tolong review". Kasih tahu persisnya APA yang harus dicek.
    Itu bedanya escalation yang berguna dan escalation yang diabaikan.
    """
    prompts: List[str] = []

    if gate.unsupported:
        listed = ", ".join(n["raw"] for n in gate.unsupported[:5])
        prompts.append(
            f"Verifikasi manual angka berikut di dokumen asli — tidak "
            f"ditemukan di potongan sumber yang diambil sistem: {listed}"
        )
    for issue in critic_issues[:3]:
        prompts.append(f"Critic menandai: {issue}")

    if risk_level == "high":
        prompts.append(
            "Tugas berisiko tinggi: cocokkan angka dengan laporan audit "
            "sebelum dipakai untuk keputusan atau dikutip."
        )
    if not prompts:
        prompts.append("Konfirmasi singkat: apakah jawaban ini menjawab pertanyaan Anda?")
    return prompts


def human_judgement_trigger(
    question: str,
    question_risk: str,
    query_tags: List[str],
    confidence: float,
    gate: NumericalGateResult,
    critic_issues: List[str],
    risk_override: Optional[str] = None,
) -> HumanTriggerResult:
    """
    Keputusan final: boleh auto-jawab, atau harus lewat manusia?

    Tiga jalur:
      auto_accept -> confidence di atas ambang, gate bersih
      review      -> di bawah ambang, tapi tidak fatal
      escalate    -> ada angka tak terdukung pada tugas berisiko tinggi
    """
    risk = assess_risk(question_risk, query_tags, risk_override)
    threshold = SOCIOTECHNICAL.confidence_threshold_by_risk.get(risk, 0.75)

    if not SOCIOTECHNICAL.enable_human_trigger:
        return HumanTriggerResult(
            needs_human=False, risk_level=risk, confidence=confidence,
            threshold=threshold, action="auto_accept",
            reasons=["Layer 4 dinonaktifkan (mode ablation)."],
        )

    reasons: List[str] = []
    needs_human = False
    action = "auto_accept"

    if confidence < threshold:
        needs_human = True
        action = "review"
        reasons.append(
            f"Confidence {confidence:.2f} di bawah ambang {threshold:.2f} "
            f"untuk tugas risiko {risk}."
        )

    if SOCIOTECHNICAL.always_escalate_on_unsupported_number and gate.unsupported:
        needs_human = True
        action = "escalate"
        reasons.append(
            f"{len(gate.unsupported)} angka dalam jawaban tidak terverifikasi "
            f"terhadap sumber."
        )

    if risk == "high" and gate.total_numbers == 0:
        # Pertanyaan minta angka tapi jawaban nol angka = kemungkinan
        # retrieval gagal, bukan kemungkinan perusahaan tidak melaporkan.
        needs_human = True
        action = "review" if action == "auto_accept" else action
        reasons.append("Tugas menuntut data numerik, tetapi jawaban tidak memuat angka.")

    if not reasons:
        reasons.append(
            f"Confidence {confidence:.2f} memenuhi ambang {threshold:.2f}; "
            f"seluruh angka terverifikasi."
        )

    return HumanTriggerResult(
        needs_human=needs_human,
        risk_level=risk,
        confidence=confidence,
        threshold=threshold,
        reasons=reasons,
        prompts_for_human=build_human_prompts(question, gate, critic_issues, risk)
        if needs_human else [],
        action=action,
    )
