"""
config.py — Semua knob VERIFIED ada di sini.

Filosofi: JANGAN hardcode apapun di file layer. Semua angka, threshold,
model name, dan pattern yang mungkin lo tune untuk eksperimen tesis
harus hidup di file ini, biar tiap perubahan eksperimen = 1 commit yang
jelas dan bisa lo tulis di Bab 4 sebagai "parameter setting".
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ==========================================================
# 1. API KEYS
# Anthropic  -> generation (console.anthropic.com, butuh prepaid credits)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Voyage AI  -> embedding, Anthropic tidak punya embedding model sendiri.
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")


def get_api_key(name: str, override: str = "") -> str:
    """Resolusi API key: argumen override > environment variable > default modul (kosong).
    Dibaca saat runtime supaya .env yang di-load setelah import tetap terpakai."""
    if override:
        return override
    val = os.getenv(name, "")
    if val:
        return val
    return {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "VOYAGE_API_KEY": VOYAGE_API_KEY}.get(name, "")

# ==========================================================
# 2. MODEL
# ==========================================================
@dataclass
class ModelConfig:
    # Model used: sonnet-5 (considered balance dibanding OPUS)
    generator_model: str = "claude-sonnet-4-6"

    # Ekstraksi entitas dari knowledge graph.
    utility_model: str = "claude-sonnet-4-6"

    # Model untuk DEBATE di Layer 3
    critic_model: str = "claude-sonnet-4-6"

    max_tokens: int = 2000

    # temperature -> uji stabilitas output (variance analysis).
    temperature: float = 0.0

    # Harga (in $ per million token), dipakai buat konversi token_count
    price_per_mtok_input: float = 3.00
    price_per_mtok_output: float = 15.00


# ==========================================================
# LAYER 1 — RETRIEVAL
@dataclass
class RetrievalConfig:
    index_dir: str = "./data/indexes"
    pdf_dir: str = os.getenv(
        "VERIFIED_PDF_DIR",
        r"C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester\Reserach Writing I\IDX30",
    )

    # Laporan tahunan IDX average > 100 pages
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Separator diurutkan dari paling "kuat" ke paling lemah.
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    # voyage-finance-2 = domain-specific finance (rekomendasi Anthropic)
    embedding_model: str = "voyage-finance-2"
    embedding_batch_size: int = 64

    # top_k besar = recall naik, tapi biaya dan context bloat naik
    top_k: int = 8

    # Ambil lebih banyak dulu, baru di-rerank/filter oleh Layer 2.
    top_k_prefetch: int = 20

# ==========================================================
# 4. LAYER 2 — CONTEXT MANAGEMENT (Regex tagging)
REGEX_TAGS: Dict[str, str] = {
    # --- Angka & mata uang (format Indonesia: 1.234.567,89) ---
    "CURRENCY": r"(?:Rp\.?\s?|IDR\s?|USD\s?|\$\s?)\d[\d.,]*(?:\s?(?:juta|miliar|milyar|triliun|ribu|bn|mn))?",
    "NUMBER": r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b|\b\d+,\d+\b",
    "PERCENTAGE": r"\b\d+(?:[.,]\d+)?\s?%",
    "YEAR": r"\b(?:19|20)\d{2}\b",

    # --- Struktur laporan ---
    "NOTE_REF": r"[Cc]atatan\s+(?:atas\s+)?(?:no\.?\s*)?\d+[a-z]?",
    "TABLE_HINT": r"(?:Tabel|Table)\s+\d+|(?:\|\s*[^|\n]+\s*){3,}\|",
    "AUDITED": r"(?:telah\s+)?diaudit|audited|opini\s+wajar|unqualified\s+opinion",

    # --- Akun keuangan inti (RQ: metrik utama) ---
    "CASHFLOW": r"arus\s+kas|cash\s*flow|kas\s+(?:neto|bersih)\s+(?:dari|yang)",
    "REVENUE": r"pendapatan|penjualan\s+neto|revenue|net\s+sales",
    "PROFIT": r"laba\s+(?:bersih|tahun\s+berjalan|kotor|usaha)|rugi\s+bersih|net\s+(?:income|profit)",
    "ASSET": r"jumlah\s+aset|total\s+aset|total\s+asset|liabilitas|ekuitas",
    "DIVIDEND": r"dividen|dividend|pembagian\s+laba|payout\s+ratio",

    # --- Tema pertanyaan riset lo ---
    "FX_RISK": r"(?:nilai\s+tukar|kurs)\s*(?:Rupiah|IDR|USD)?|volatilitas\s+kurs|foreign\s+exchange|selisih\s+kurs",
    "ENERGY_COST": r"biaya\s+energi|harga\s+(?:bahan\s+bakar|batu\s?bara|gas|listrik)|energy\s+cost",
    "SEGMENT": r"segmen\s+(?:usaha|operasi|bisnis)|business\s+segment|lini\s+bisnis|entitas\s+anak|anak\s+perusahaan",
    "ESG": r"\bESG\b|keberlanjutan|sustainability|emisi\s+(?:karbon|GRK)|net\s*zero|tanggung\s+jawab\s+sosial",
    "RISK_MITIGATION": r"mitigasi\s+risiko|manajemen\s+risiko|lindung\s+nilai|hedg(?:e|ing)|strategi\s+(?:jangka|mitigasi)",
}


@dataclass
class ContextConfig:
    # Semua dimanage di app oleh user jadi bisa lebih mudah menggunakan
    max_context_tokens: int = 3000
    max_chunks: int = 6
    weight_similarity: float = 0.7
    weight_tag_match: float = 0.3
    # Bobot skor: skor_akhir = (w_sim * similarity) + (w_tag * tag_overlap)

    # Anti lost-in-the-middle
    # chunk paling relevan ditaruh di AWAL dan AKHIR context
    reorder_lost_in_the_middle: bool = True

    # Estimasi kasar: 1 token ~ 4 karakter
    chars_per_token: float = 3.5
    # atau pakai client.messages.count_tokens()).


# ==========================================================
# LAYER 3 — VERIFICATION (numerical gate + DEBATE)
# ==========================================================
@dataclass
class VerificationConfig:
    enable_numerical_gate: bool = True
    enable_adversarial_critique: bool = True

    # Toleransi relatif saat mencocokkan angka jawaban vs angka di sumber.
    # 0.0 = harus persis. 0.01 = toleransi 1% (buat pembulatan).
    numeric_tolerance: float = 0.01

    # Berapa ronde Scorer -> Critic -> Commander.
    # 1 ronde cukup untuk tesis; 2 buat uji marginal gain (bahan Bab 5).
    debate_rounds: int = 1

    # Kalau ada angka di jawaban yang TIDAK ketemu di context,
    # confidence dipotong sebanyak ini per angka bermasalah.
    unsupported_number_penalty: float = 0.10

    # Di bawah ini jawaban ditandai LOW_CONFIDENCE.
    min_confidence: float = 0.70


# ==========================================================
# 6. LAYER 4 — SOCIOTECHNICAL (human judgement trigger)
# ==========================================================
@dataclass
class SociotechnicalConfig:
    enable_human_trigger: bool = True

    # Matriks inti Layer 4: makin tinggi risiko tugas, makin tinggi
    # confidence yang dibutuhkan sistem untuk boleh menjawab sendiri.
    # Ini operasionalisasi "beban verifikasi manusia diselaraskan dengan
    # tingkat risiko tugas".
    confidence_threshold_by_risk: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.55,     # deskriptif: "segmen apa saja yang disebut?"
            "medium": 0.75,  # interpretatif: "apakah cashflow positif?"
            "high": 0.90,    # angka absolut / forward-looking / dividen
        }
    )

    # Tag yang otomatis menaikkan risk level ke "high".
    # Rasional: kesalahan angka di laporan keuangan
    high_risk_tags: List[str] = field(
        default_factory=lambda: ["CURRENCY", "PERCENTAGE", "DIVIDEND", "PROFIT"]
    )
    medium_risk_tags: List[str] = field(
        default_factory=lambda: ["CASHFLOW", "FX_RISK", "RISK_MITIGATION", "ASSET"]
    )

    # Kalau jawaban mengandung angka yang tak terdukung, LANGSUNG eskalasi ke manusia
    always_escalate_on_unsupported_number: bool = True


# ==========================================================
# 7. EKSPERIMEN
# ==========================================================
# 8 pertanyaan dari Tabel 3.2 tesis (adopsi Spörer, 2025).
# risk = level risiko awal; bisa dinaikkan otomatis oleh Layer 4.
EVAL_QUESTIONS: List[Dict[str, str]] = [
    {"id": "Q1", "risk": "high",
     "text": "Apa saja angka dan metrik keuangan utama yang disebutkan dalam laporan ini?"},
    {"id": "Q2", "risk": "high",
     "text": "Informasi apa yang diberikan mengenai arus kas perusahaan? Apakah arus kas perusahaan positif atau negatif pada akhir tahun buku laporan ini?"},
    {"id": "Q3", "risk": "medium",
     "text": "Apakah laporan menyebut dampak volatilitas kurs Rupiah/USD atau kenaikan biaya energi terhadap arus kas atau modal kerja?"},
    {"id": "Q4", "risk": "low",
     "text": "Segmen bisnis atau divisi apa saja yang disebutkan di laporan, dan apakah ada lini bisnis baru yang diumumkan?"},
    {"id": "Q5", "risk": "medium",
     "text": "Apakah perusahaan menyebut strategi mitigasi risiko nilai tukar atau biaya energi dalam rencana jangka pendek-menengah?"},
    {"id": "Q6", "risk": "low",
     "text": "Apa saja bisnis yang dimiliki di bawah perusahaan terkait dan bergerak di bidang apa saja?"},
    {"id": "Q7", "risk": "high",
     "text": "Apakah perusahaan sudah membagi dividen pada tahun terkait, dan berapa nominal yang didistribusikan bagi pemegang saham?"},
    {"id": "Q8", "risk": "medium",
     "text": "Apakah perusahaan menyebutkan komitmennya dalam ESG secara spesifik dalam laporan tahunan? Sertakan angka kuantitatif yang ditetapkan apabila ada."},
]

SCENARIOS = ["vanilla", "standard_rag", "verified"]

RESULTS_DIR = "./results"
LOG_FILE = "./results/run_log.csv"


# 8. INSTANCE GLOBAL — import ini di file lain
# ==========================================================
MODEL = ModelConfig()
RETRIEVAL = RetrievalConfig()
CONTEXT = ContextConfig()
VERIFICATION = VerificationConfig()
SOCIOTECHNICAL = SociotechnicalConfig()
