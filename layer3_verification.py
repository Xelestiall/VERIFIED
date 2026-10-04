"""
layer3_verification.py — LAYER 3: VERIFICATION
  A. NUMERICAL VERIFICATION GATE (deterministik, tanpa LLM)
     Setiap angka di jawaban wajib ada di chunk sumber. Ini gate
     berbasis aturan, bukan LLM — jadi tidak bisa ikut berhalusinasi.
     Ini yang menyerang langsung "factual fabrication" (Huang et al., 2025).

  B. ADVERSARIAL CRITIQUE (adopsi DEBATE, Kim et al., 2024)
     Scorer    -> nilai jawaban awal, keluarkan confidence
     Critic    -> peran devil's advocate, cari cacat secara agresif
     Commander -> putuskan revisi final + confidence final
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from config import MODEL, VERIFICATION
from layer1_retrieval import Chunk

# ==========================================================
# A. NUMERICAL VERIFICATION GATE
MAGNITUDE = {
    "ribu": 1e3, "juta": 1e6, "miliar": 1e9, "milyar": 1e9,
    "triliun": 1e12, "trilyun": 1e12,
    "k": 1e3, "mn": 1e6, "bn": 1e9, "tn": 1e12,
}


NUMBER_RX = re.compile(
    # Urutan alternatif PENTING: format Inggris (koma = ribuan) dicoba lebih dulu.
    # Beberapa laporan IDX30 (mis. BUMI, AADI pada catatan 'full amount') memakai
    # "371,335,392,068" walau berbahasa Indonesia; tanpa ini terpotong jadi
    # "371,335" dan "392,068" lalu dibaca sebagai desimal -> angka salah.
    r"(?<![\d.,])(\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?|\d{1,3},\d{3}\.\d+"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?|\d+(?:\.\d+)?)"
    # (?![A-Za-z]) mencegah satuan menempel pada kata: "3 kali" bukan 3k, "5 tn" ok.
    r"\s*(ribu|juta|miliar|milyar|triliun|trilyun|k|mn|bn|tn|%)?(?![A-Za-z])",
    re.IGNORECASE,
)

# Penomoran daftar ("1. ", "2) ", "- 3. ") di awal baris BUKAN angka keuangan.
# Tanpa ini, jawaban berformat daftar menambah angka palsu ke penyebut gate.
LIST_MARKER_RX = re.compile(r"(?m)^[ \t]*(?:[-*\u2022][ \t]*)?(?:\*\*)?\d{1,2}[.)](?=\s)")

# Header satuan tabel laporan IDX: "dalam jutaan Rupiah" / "in millions of Rupiah".
UNIT_HEADER_RX = re.compile(
    r"\b(?:dalam|dinyatakan\s+dalam|disajikan\s+dalam|in|expressed\s+in|stated\s+in|presented\s+in)\s+"
    r"(jutaan|juta|ribuan|ribu|miliaran|miliar|millions?|thousands?|billions?)\b",
    re.IGNORECASE,
)
HEADER_SCALE = {
    "jutaan": 1e6, "juta": 1e6, "million": 1e6, "millions": 1e6,
    "ribuan": 1e3, "ribu": 1e3, "thousand": 1e3, "thousands": 1e3,
    "miliaran": 1e9, "miliar": 1e9, "billion": 1e9, "billions": 1e9,
}


def detect_table_scale(text: str) -> float:
    """Skala satuan tabel pada sebuah chunk (1.0 bila tidak ada header satuan).

    Laporan keuangan IDX menyajikan angka tabel 'dalam jutaan Rupiah': sel
    bernilai 3.301.470 artinya Rp3,3 triliun. Gate versi awal membandingkan
    '3.301.470 juta' (=3,3e12) dengan 3.301.470 (=3,3e6) lalu menandainya
    tak-terdukung -> SEMUA angka tabel jadi false positive dan Layer 4
    mengeskalasi hampir setiap jawaban. Ini membetulkannya.
    """
    m = UNIT_HEADER_RX.search(text)
    return HEADER_SCALE.get(m.group(1).lower(), 1.0) if m else 1.0

# Konsekuensi dari melonggarkan lookbehind, hallucination rate jadi kotor.
CITATION_RX = re.compile(r"\[([^\[\]]+?)::c(\d+)\]")

def parse_id_number(raw: str, unit: str = "") -> Optional[float]:
    """Ubah string angka gaya Indonesia jadi float.
      '1.234.567,89'      -> 1234567.89
      '4,7' + 'triliun'   -> 4.7e12
      '12,5' (%)          -> 12.5
      '1.234' bisa berarti seribu dua ratus tiga puluh empat (Indonesia) ATAU 1,234 desimal (Inggris).    
    Laporan IDX konsisten pakai konvensi Indonesia
    """
    s = raw.strip()
    try:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s) and (s.count(",") >= 2 or "." in s):
            value = float(s.replace(",", ""))   # format Inggris: koma = ribuan
        elif "," in s and "." in s:
            value = float(s.replace(".", "").replace(",", "."))
        elif "," in s:
            value = float(s.replace(",", "."))
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            value = float(s.replace(".", ""))
        else:
            value = float(s)
    except ValueError:
        return None

    u = unit.lower().strip()
    if u in MAGNITUDE:
        value *= MAGNITUDE[u]
    return value

def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Ambil semua angka + nilai ternormalisasi dari sepotong teks.

    Field yang dikembalikan:
      raw, value, is_percent,
      is_year     -> 4 digit 1900-2100 tanpa satuan/pemisah ribuan (dikecualikan gate)
      has_unit    -> ada satuan skala (juta/miliar/...) menempel pada angka
      decimals    -> jumlah digit desimal yang DITAMPILKAN (dasar toleransi pembulatan)
      unit_scale  -> pengali satuan (1.0 bila tanpa satuan)
    """
    text = CITATION_RX.sub(" ", text)
    text = LIST_MARKER_RX.sub(" ", text)
    out: List[Dict[str, Any]] = []
    for m in NUMBER_RX.finditer(text):
        num_str, unit = m.group(1), (m.group(2) or "")
        val = parse_id_number(num_str, "" if unit == "%" else unit)
        if val is None:
            continue
        u = unit.lower()
        is_pct = u == "%"
        has_unit = (not is_pct) and u in MAGNITUDE
        is_year = (
            not unit
            and re.fullmatch(r"(?:19|20)\d{2}", num_str) is not None
        )
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", num_str) and (num_str.count(",") >= 2 or "." in num_str):
            decimals = len(num_str.split(".")[1]) if "." in num_str else 0   # format Inggris
        else:
            decimals = len(num_str.split(",")[1]) if "," in num_str else 0
        out.append({
            "raw": m.group(0).strip(),
            "value": val,
            "is_percent": is_pct,
            "is_year": is_year,
            "has_unit": has_unit,
            "decimals": decimals,
            "unit_scale": MAGNITUDE.get(u, 1.0) if has_unit else 1.0,
        })
    return out

@dataclass
class NumericalGateResult:
    passed: bool
    total_numbers: int
    verified_numbers: int
    unsupported: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def support_ratio(self) -> float:
        return self.verified_numbers / self.total_numbers if self.total_numbers else 1.0

def _rounding_tolerance(n: Dict[str, Any]) -> float:
    """Setengah satuan digit terakhir yang ditampilkan di jawaban.

    'Rp3,3 triliun' menampilkan 1 desimal pada skala 1e12 -> ±0,05 triliun.
    'Rp3.301.470 juta' menampilkan 0 desimal pada skala 1e6 -> ±0,5 juta.
    Angka yang ditampilkan presisi tidak boleh lolos hanya karena 'dekat'
    dengan angka lain di konteks.
    """
    return 0.5 * (10.0 ** -n["decimals"]) * n["unit_scale"]


def numerical_gate(
    answer: str,
    chunks: List[Chunk],
    extra_scales: Optional[Iterable[float]] = None,
) -> NumericalGateResult:
    """
    Cocokkan tiap angka di jawaban dengan angka di chunk sumber.

    Pencocokan memakai NILAI, bukan string, dan sadar-skala-tabel:
      'Rp3,3 triliun' di jawaban cocok dengan sel '3.301.470' pada tabel
      berheader 'dalam jutaan Rupiah'.
    Toleransi = min(pembulatan-yang-ditampilkan, VERIFICATION.numeric_tolerance).
    Jadi numeric_tolerance (1%) berfungsi sebagai BATAS ATAS, bukan toleransi
    datar; angka presisi 7 digit tidak lolos hanya karena meleset 0,5%.

    Catatan keterbatasan (tulis di Bab 5): gate memeriksa EKSISTENSI nilai di
    konteks, bukan bahwa nilai itu milik akun yang ditanyakan. Jawaban yang
    menukar 'laba bruto' dengan 'laba bersih' akan lolos. Selisih ini harus
    ditangkap anotasi manusia, dan justru menjadi temuan tentang batas gate.
    """
    if not VERIFICATION.enable_numerical_gate:
        return NumericalGateResult(passed=True, total_numbers=0, verified_numbers=0)

    # Header "dalam jutaan Rupiah" hanya muncul di chunk pembuka tabel; chunk
    # badan tabel yang ikut terambil sering TIDAK memuatnya (pada KLBF: 136 dari
    # 661 chunk berheader). Maka skala dikumpulkan dari seluruh chunk terambil
    # DAN dari `extra_scales` (skala tingkat-dokumen, RetrievalLayer.table_scales()).
    scales = {detect_table_scale(c.text) for c in chunks}
    scales.update(extra_scales or [])
    scales.discard(1.0)

    source_numbers: List[float] = []
    for c in chunks:
        for n in extract_numbers(c.text):
            source_numbers.append(n["value"])
            # Sel tabel tanpa satuan eksplisit mewarisi skala header tabel.
            if not n["has_unit"] and not n["is_percent"] and not n["is_year"]:
                source_numbers.extend(n["value"] * sc for sc in scales)

    answer_numbers = [n for n in extract_numbers(answer) if not n["is_year"]]
    cap = VERIFICATION.numeric_tolerance

    unsupported = []
    verified = 0
    for n in answer_numbers:
        val = n["value"]
        match = any(
            abs(val - sv) <= min(_rounding_tolerance(n), cap * max(abs(val), abs(sv), 1.0))
            for sv in source_numbers
        )
        if match:
            verified += 1
        else:
            unsupported.append(n)

    # Penalti deterministik DINONAKTIFKAN (permintaan): confidence murni dari
    # penilaian Scorer/Commander atas sumber tersitasi, bukan potongan mekanis
    # per angka. Gate tetap menghitung `unsupported` untuk laporan & Layer 4.
    return NumericalGateResult(
        passed=len(unsupported) == 0,
        total_numbers=len(answer_numbers),
        verified_numbers=verified,
        unsupported=unsupported
    )


# ==========================================================
# B. ADVERSARIAL CRITIQUE — DEBATE
# ==========================================================
SCORER_SYSTEM = """Anda adalah Scorer dalam sistem verifikasi jawaban laporan keuangan.
Tugas Anda menilai apakah jawaban benar-benar didukung oleh sumber yang diberikan.
Anda TIDAK menilai gaya bahasa. Anda hanya menilai kesetiaan terhadap sumber (faithfulness)
dan kebenaran faktual."""

CRITIC_SYSTEM = """Anda adalah Critic yang berperan sebagai devil's advocate.
Tugas Anda MENYERANG jawaban, bukan membelanya. Asumsikan jawaban itu salah
sampai terbukti benar. Cari: angka yang tidak ada di sumber, klaim yang
melampaui isi dokumen, instruksi pertanyaan yang tidak dijawab, dan
kesimpulan yang ditarik tanpa dasar.
Jika setelah diperiksa jawaban memang solid, katakan demikian secara singkat.
Jangan mengarang cacat yang tidak ada."""

COMMANDER_SYSTEM = """Anda adalah Commander. Anda menerima jawaban awal, penilaian Scorer,
dan serangan Critic. Tugas Anda membuat keputusan final: pertahankan, revisi, atau
tandai sebagai tidak dapat dijawab dari sumber yang tersedia.
Prinsip: lebih baik mengatakan "tidak ditemukan dalam dokumen" daripada menebak."""


@dataclass

class DebateResult:
    final_answer: str
    confidence: float
    scorer_confidence: float = 0.0
    critic_issues: List[str] = field(default_factory=list)
    revised: bool = False
    rounds: int = 0
    verdict: str = "kept"

def run_debate(
    llm: Any,
    question: str,
    draft_answer: str,
    context: str,
    gate: NumericalGateResult,
    on_step: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
) -> DebateResult:
    """Jalankan siklus Scorer -> Critic -> Commander."""
    if not VERIFICATION.enable_adversarial_critique:
        return DebateResult(
            final_answer=draft_answer,
            confidence=max(0.0, 1.0),
            verdict = "kept",
        )

    answer = draft_answer
    scorer_conf = 0.0
    issues: List[str] = []
    verdict = "kept"
    revised = False

    gate_report = "Semua angka dalam jawaban terverifikasi di sumber."
    if gate.unsupported:
        listed = ", ".join(n["raw"] for n in gate.unsupported[:8])
        gate_report = (
            f"PERINGATAN GATE NUMERIK: {len(gate.unsupported)} angka TIDAK "
            f"ditemukan di sumber manapun: {listed}"
        )

    for rnd in range(VERIFICATION.debate_rounds):
        # ---- Scorer ----
        if on_step:
            on_step("scorer", None)
        scored = llm.complete_json(
            f"PERTANYAAN:\n{question}\n\n"
            f"SUMBER:\n{context}\n\n"
            f"JAWABAN YANG DINILAI:\n{answer}\n\n"
            f"HASIL PEMERIKSAAN NUMERIK OTOMATIS:\n{gate_report}\n\n"
            'Nilai jawaban. Format: {"confidence":0.0-1.0,'
            '"supported_claims":["..."],"unsupported_claims":["..."],'
            '"reasoning":"ringkas, maksimal 2 kalimat"}',
            system=SCORER_SYSTEM,
            model=MODEL.generator_model,
            stage="scorer",
            fallback={"confidence": 0.5, "unsupported_claims": [], "reasoning": ""},
        )
        scorer_conf = float(scored.get("confidence", 0.5))
        if on_step:
            on_step("scorer", {
                "confidence": scorer_conf,
                "reasoning": scored.get("reasoning", ""),
                "supported": scored.get("supported_claims", []),
                "unsupported": scored.get("unsupported_claims", []),
            })

        # ---- Critic ----
        if on_step:
            on_step("critic", None)
        critique = llm.complete_json(
            f"Question: \n{question}\n\n"
            f"Context: \n{context}\n\n"
            f"Answer: \n{answer}\n\n"
            'Format: {"issues":["..."],"severity":"none|low|medium|high",'
            '"suggested_fix":"..."}',
            system=CRITIC_SYSTEM,
            model=MODEL.critic_model,
            stage="critic",
            fallback={"issues": [], "severity": "none", "suggested_fix": ""},
        )
        round_issues = critique.get("issues", []) or []
        issues.extend(round_issues)
        severity = str(critique.get("severity", "none")).lower()
        if on_step:
            on_step("critic", {
                "issues": round_issues,
                "severity": severity,
                "suggested_fix": critique.get("suggested_fix", ""),
            })

        # Kalau Critic tidak menemukan apa-apa dan gate lolos, berhenti.
        if severity in ("none", "low") and gate.passed:
            break

        # ---- Commander ----
        if on_step:
            on_step("commander", None)
        decision = llm.complete_json(
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Confidence={scorer_conf}; {scored.get('reasoning','')}\n\n"
            f"Critic ({severity}): {round_issues}\n"
            f"Fix: {critique.get('suggested_fix','')}\n\n"
            'Format: {"final_answer":"...","confidence":0.0-1.0,'
            '"verdict":"kept|revised|not_answerable"}',
            system=COMMANDER_SYSTEM,
            model=MODEL.generator_model,
            stage="commander",
            fallback={"final_answer": answer, "confidence": scorer_conf, "verdict": "kept"},
        )
        new_answer = decision.get("final_answer", answer)
        if new_answer and new_answer != answer:
            revised = True
            answer = new_answer
        scorer_conf = float(decision.get("confidence", scorer_conf))
        verdict = decision.get("verdict", "revised" if revised else "kept")
        if on_step:
            on_step("commander", {
                "verdict": verdict,
                "confidence": scorer_conf,
                "revised": revised,
                "suggested_fix": critique.get("suggested_fix", ""),
            })

    # Confidence Score final
    confidence = max(0.0, min(1.0, scorer_conf))

    return DebateResult(
        final_answer=answer,
        confidence=round(confidence, 3),
        scorer_confidence=round(scorer_conf, 3),
        critic_issues=issues,
        revised=revised,
        rounds=VERIFICATION.debate_rounds,
        verdict=verdict,
    )
