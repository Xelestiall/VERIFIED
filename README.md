# VERIFIED Framework

Implementasi framework VERIFIED — sociotechnical context engineering melalui confidence score
untuk mengurangi halusinasi AI pada analisis laporan keuangan tahunan IDX30 Periode 1 Agustus - 31 Oktober 2026.
---

## Peta file → layer tesis
| File                      | Isi                                                                               |
| `config.py`               | Configurasi Parameter yang bisa diset ulang                                       |
| `llm.py`                  | Wrapper Anthropic + token ledger (formula `Cost(s)`)                              |
| `layer1_retrieval.py`     | RAG menggunakan LangChain, Embedding Voyage, Knowledge Graph NetworkX, FAISS      |
| `layer2_context.py`       | Context management melalui Regex tagging, untuk antitesa lost-in-the-middle       |
| `layer3_verification.py`  | Numerical gate + DEBATE (Scorer/Critic/Commander)                                 |
| `layer4_sociotechnical.py`| *Human judgement trigger* (risk × confidence)                                     |
| `pipeline.py`             | Perakitan + 3 skenario eksperimen                                                 |
| `runner.py`               | Batch runner 30 emiten × 3 skenario × 8 soal                                      |
| `evaluation.py`           | HR, token, traceability, RAGAS                                                    |
| `app.py`                  | Artefak Streamlit untuk Interface  & Development yang bisa digunakan oleh user    |

---

## Setup
### 1. Dua API key, dua akun terpisah
*Anthropic    -> API Key LLM
*Voyage AI    -> API Key embedding model

### 2. Install
```bash
pip install -r requirements.txt
mkdir -p data/pdf data/indexes results --a
```

### 3. Smoke Test
```bash
# 1 PDF, 1 skenario, 2 soal. Sekitar 30 detik, biaya < $0.05.
python runner.py --pdf data/pdf/BBCA_2025.pdf --scenarios verified --questions Q1 Q2
python runner.py --pdf-dir ./data/pdf --limit 3           # pilot 3 emiten
python runner.py --pdf-dir ./data/pdf --resume            # full run, aman diputus
python evaluation.py --csv results/run_log.csv            # hitung metrik
streamlit run app.py                                       # artefak interaktif
```
---

## Alur data
```
PDF → [L1] chunk + embed + FAISS ──┐
                                   ├─→ prefetch 20 chunk
      [L2] tag → rerank → reorder → 6 chunk
                                   ↓
                            generate draft
                                   ↓
      [L3a] numerical gate (deterministik, tanpa LLM)
      [L3b] Scorer → Critic → Commander
                                   ↓
      [L4] risk × confidence → auto / review / escalate
                                   ↓
                     jawaban + sitasi + token log
```

Perhatikan: prefetch 20, kirim 6. Standard RAG ambil 8, kirim 8.
Jadi VERIFIED punya jangkauan *recall* lebih luas tapi *context* lebih ramping.

---
## To be noted
**1. Hallucination rate butuh anotasi manusia.** `evaluation.py` menghitung Hallucination Rate dari kolom `is_hallucination` yang 
diisi manual di CSV. Ada proxy otomatis (`unsupported_number_pct`), 
Sistem menilai dirinya sendiri pakai gate yang dia sendiri jalankan, dan penguji berhak menolak. 
Pakai proxy untuk memprioritaskan baris mana yang dianotasi duluan.

Untuk reliabilitas anotasi, idealnya ada 2 annotator pada subset (misal 20%) lalu hitung Cohen's Kappa. Spörer (2025) yang lo kritik dapat κ = 0,37 — kalau lo bisa lebih tinggi, itu kontribusi metodologis tersendiri.

**2. Context recall butuh ground truth.** Lo harus menulis jawaban acuan manual dari laporan asli untuk tiap (emiten, pertanyaan). 30 × 8 = 240 jawaban acuan. Alokasikan waktu untuk ini, atau kurangi sampel untuk dimensi recall saja dan sebutkan di limitasi.

**3. `temperature=0` wajib untuk run resmi.** Kalau tidak, hasil lo tidak reproducible dan itu titik serang paling gampang di sidang.

**4. Ablation study hampir gratis.** Semua layer bisa dimatikan lewat `config.py`:
| Setting | Menguji |
| `weight_tag_match = 0.0` | Kontribusi regex tagging |
| `enable_numerical_gate = False` | Kontribusi numerical gate |
| `enable_adversarial_critique = False` | Kontribusi DEBATE |
| `reorder_lost_in_the_middle = False` | Kontribusi anti-LITM |
| `embedding_provider = "local"` | Kontribusi embedding domain-finance |

Lima baris tabel di Bab 4, tiap baris satu run. Ini yang mengubah klaim "VERIFIED bekerja" jadi "ini bagian mana yang bekerja" — jauh lebih kuat.
---

## Estimasi biaya
Per pertanyaan mode VERIFIED: ~4 panggilan LLM (generate + scorer + critic + commander), sekitar 8–15k token total.
| Cakupan | Perkiraan |
|---|---|
| 1 emiten × 8 soal × 3 skenario | ~$0,40 |
| 30 emiten (full) | ~$12 |
| + embedding Voyage 30 PDF | ~$2–4 |
| + RAGAS evaluation | ~$5–8 |

Kasar dan tergantung panjang PDF, tapi ordenya puluhan dolar, bukan ratusan. Index FAISS di-cache ke disk, jadi run ulang tidak membayar embedding dua kali.
