# VERIFIED — Technical Specification
Dokumen ini disegarkan pada 5 Okt 2026 dengan membaca kode sumber aktual di commit `59a6bd9` (branch `main`). Versi sebelumnya disusun dari commit `6cd1156`, sebelum perbaikan 4 Okt (`d2e8417`) dan penambahan buffer sesi untuk baseline (`59a6bd9`); ringkasan perbedaannya ada di bagian (f).
Setiap klaim disertai referensi `file:baris`. Nomor baris hanya valid untuk commit di atas dan akan bergeser kalau kode berubah. Bagian yang tidak bisa dipastikan dari kode ditandai **[BELUM DITENTUKAN]**.

---

## (a) Arsitektur per layer

| Layer | File implementasi | Isi |
|---    |---                |---  |
| **Layer 1 — Retrieval** | `layer1_retrieval.py` | `Chunk` (struktur data), `VoyageEmbeddings` / `LocalEmbeddings` (adapter embedding), `get_embeddings()` (factory), `RetrievalLayer` (ingest PDF → chunk → embed → FAISS, plus `search()` dan `table_scales()` — skala satuan tabel tingkat-dokumen untuk gate numerik), `KnowledgeGraph` (ekstraksi entitas opsional via NetworkX) |
| **Layer 2 — Context Management** | `layer2_context.py` | `tag_text`/`tag_chunks`/`tag_query` (regex tagging), `tag_overlap_score`/`rerank` (skor gabungan similarity+tag), `apply_budget` (potong sesuai token budget), `reorder_against_lost_in_the_middle` (susun ulang anti-LITM), `ConversationBuffer` (riwayat percakapan, tanpa limit turn), `build_context` (orkestrasi keempatnya → `ManagedContext`) |
| **Layer 3 — Verification** | `layer3_verification.py` | Bagian A: `extract_numbers`/`parse_id_number`/`detect_table_scale`/`numerical_gate` (gate numerik deterministik, non-LLM, sadar-skala-tabel). Bagian B: `run_debate` — siklus Scorer → Critic → Commander (`SCORER_SYSTEM`, `CRITIC_SYSTEM`, `COMMANDER_SYSTEM`) |
| **Layer 4 — Sociotechnical / Human Review** | `human_review.py` | `assess_risk` (naikkan risk level berdasar tag), `build_human_prompts` (susun instruksi konkret utk analis), `human_judgement_trigger` (putuskan `auto_accept` / `review` / `escalate`) |
| **Orkestrasi + 3 skenario eksperimen** | `pipeline.py` | `VanillaPipeline` (LLM murni, tanpa dokumen), `StandardRAGPipeline` (Layer 1 saja), `VerifiedPipeline` (Layer 1+2+3+4); ketiganya kini punya `ConversationBuffer` dan parameter `use_buffer`. Juga `PipelineResult` (skema output flat, siap jadi baris CSV), `build_pipeline()` (factory), `extract_citations()` |
| **LLM wrapper + token accounting** | `llm.py` | `LLMClient` (satu-satunya titik panggil Anthropic SDK: `.complete()`, `.complete_json()`, `.count_tokens()`), `TokenLedger` (akumulasi token per-stage + biaya), `safe_json_parse()` |
| **Konfigurasi terpusat** | `config.py` | Semua dataclass parameter (`ModelConfig`, `RetrievalConfig`, `ContextConfig`, `VerificationConfig`, `SociotechnicalConfig`), `REGEX_TAGS`, `EVAL_QUESTIONS`, API-key loader `get_api_key()` |
| **Sumber dokumen (multi-backend)** | `sources.py` | `BaseSource` + implementasi `LocalSource`, `SupabaseSource`, `GoogleDriveSource`, `SharePointSource`; `make_doc_id()` (kunci join antar CSV); `build_source()` (factory dari CLI args) |
| **Batch runner eksperimen** | `runner.py` | `ExperimentRunner` (loop dokumen × skenario × soal; opsi `--session-buffer`), `ResultLogger` (tulis `results/run_log.csv`, dukung `--resume`; `write_build_cost()` menulis biaya sekali-bangun ke `results/build_costs.csv`). Memuat `.env` lewat `load_dotenv()` (`runner.py:21-22`) |
| **Evaluasi/metrik** | `evaluation.py` | Hallucination rate (dari anotasi manual `is_hallucination`), proxy otomatis, analisis token, traceability, human-trigger analysis, "Opsi B" (`run_judge_confidence` — judge post-hoc terpisah dari confidence bawaan pipeline), integrasi RAGAS opsional. `load_results()` menambah kolom turunan `ticker` untuk join antar skenario |
| **UI interaktif** | `app.py` | Streamlit — upload PDF, pilih skenario, jalankan pipeline, tampilkan jawaban + confidence + expander debate live + tab Sumber/Konteks/Biaya |
| **Persiapan data sumber** | `scripts/link_pdfs.py` | Dedup PDF via SHA-256, hardlink/symlink/copy ke `./data/pdf`, exclude file non-laporan |
| **Tes regresi** | `tests/test_numerical_gate.py`, `tests/test_session_buffer.py` | 12 tes gate numerik + 5 tes buffer sesi baseline (LLM palsu, tanpa API). Jalankan: `python -m pytest tests -q` (17 tes) |

---

## (b) Tabel parameter aktual

Nilai diambil dari `config.py` (sumber tunggal default) dengan catatan di kolom "Runtime" bila nilainya bisa berubah saat dijalankan (Streamlit sidebar / CLI flag). Kalau tidak dicatat, nilai **tetap** — tidak ada jalur di kode yang mengubahnya.

### Model (`ModelConfig`, `config.py:25-42`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `generator_model` | `"claude-sonnet-4-6"` | Bisa diganti di sidebar `app.py:100-102` — pilihan: `"claude-sonnet-4-6"` atau `"claude-opus-4-8"`. Nilai literal ini, apa adanya dari kode; belum diverifikasi terhadap katalog model Anthropic yang berlaku saat dokumen ini ditulis. |
| `critic_model` | `"claude-sonnet-4-6"` | Di `app.py:103` **dipaksa sama dengan** `generator_model` setiap render sidebar (`MODEL.critic_model = MODEL.generator_model`) — di jalur Streamlit, ketiga model role selalu identik. Di `runner.py` (batch, tanpa Streamlit) nilai ini independen dari config, tapi kebetulan default-nya sama. |
| `utility_model` | `"claude-sonnet-4-6"` | Sama seperti `critic_model` — dipaksa sama dengan `generator_model` di `app.py:104`. Dipakai untuk ekstraksi Knowledge Graph (`layer1_retrieval.py:295`). |
| `max_tokens` | `10000` | Tidak pernah di-override per-call di kode manapun (`llm.py:120`) — berlaku untuk semua panggilan (generation, scorer, critic, commander, KG extraction). |
| `temperature` | `0.0` | Tidak pernah di-override per-call di kode manapun (`llm.py:121`). |
| `price_per_mtok_input` | `3.00` (USD/juta token) | Tetap. |
| `price_per_mtok_output` | `15.00` (USD/juta token) | Tetap. |

### Layer 1 — Retrieval (`RetrievalConfig`, `config.py:48-72`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `index_dir` | `"./data/indexes"` | Di-override oleh `app.py` per-sesi (folder temp OS, lihat `get_session_storage()` — `app.py:53-61`). `runner.py` pakai nilai default ini. |
| `pdf_dir` | env `VERIFIED_PDF_DIR`, fallback path lokal `C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester\Reserach Writing I\IDX30` | **Tidak dipakai di manapun** selain didefinisikan (lihat bagian (e)) — `runner.py` punya argumen `--pdf-dir` sendiri (default `"./data/pdf"`, `runner.py:215`) yang tidak membaca field ini. |
| `chunk_size` | `1000` (karakter) | Tetap — dipakai `RecursiveCharacterTextSplitter` (`layer1_retrieval.py:152`). |
| `chunk_overlap` | `200` (karakter) | Tetap. |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Tetap. |
| `embedding_model` | `"voyage-finance-2"` | Tetap (default `VoyageEmbeddings`). |
| `embedding_batch_size` | `64` | Tetap. |
| `top_k` | `8` | Dipakai `StandardRAGPipeline` (`pipeline.py:154`, lewat `RetrievalLayer.search()` default) dan sebagai default umum `RetrievalLayer.search()` (`layer1_retrieval.py:225`). |
| `top_k_prefetch` | `20` | Dipakai HANYA oleh `VerifiedPipeline` (`pipeline.py:225`) untuk prefetch sebelum Layer 2 memangkas ke `max_chunks`. |
| **Embedding provider** | `"voyage"` | Default parameter fungsi (`layer1_retrieval.py:110,132`, `app.py:173` eksplisit `"voyage"`); alternatif `"local"` (`SentenceTransformer` `intfloat/multilingual-e5-base`, `layer1_retrieval.py:94`) hanya lewat `runner.py --embedding local` (`runner.py:225`) — tidak ada opsi ganti provider di UI Streamlit. |
| **Vector store** | FAISS lokal (`langchain_community.vectorstores.FAISS`, via `.from_texts` / `.save_local` / `.load_local`) | Jarak diukur L2 (dinyatakan eksplisit di docstring `layer1_retrieval.py:219`), dikonversi ke similarity via `1/(1+dist)` (`layer1_retrieval.py:240`). Kelas index FAISS persis yang dipilih `langchain_community` di baliknya: **[BELUM DITENTUKAN]** (tidak diset eksplisit di repo ini). |

### Layer 2 — Context Management (`ContextConfig` + `REGEX_TAGS`, `config.py:76-118`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `max_context_tokens` | `3000` | Bisa diubah slider `app.py:123-129` (1000–10000, step 250) — hanya berlaku saat skenario `verified` dipilih (slider `disabled` untuk skenario lain, tapi field global tetap berubah). Catatan: budget ini dipakai `apply_budget()` pada **chunk saja**; riwayat percakapan ditambahkan setelah budget (`layer2_context.py:204-205`) sehingga tidak ikut dihitung. |
| `max_chunks` | `6` | Tetap — tidak ada kontrol UI. |
| `weight_similarity` | `0.7` | Bisa diubah slider `app.py:112-120` (0.0–1.0, step 0.05). |
| `weight_tag_match` | `0.3` | Turunan otomatis: `1 - weight_similarity` (dibulatkan 2 desimal, `app.py:121`), bukan slider independen. |
| `reorder_lost_in_the_middle` | `True` | Tetap — tidak ada kontrol UI, hanya lewat edit `config.py` langsung (jalur ablation manual). Dipakai `reorder_against_lost_in_the_middle()` (`layer2_context.py:117-129`). Terpisah dari `ConversationBuffer`. |
| `chars_per_token` | `3.5` | Tetap — dipakai estimasi cepat token (`layer2_context.py:100`, fallback `llm.py:186`) saat `count_tokens()` API gagal. |
| `REGEX_TAGS` | **17** tag: `CURRENCY, NUMBER, PERCENTAGE, YEAR, NOTE_REF, TABLE_HINT, AUDITED, CASHFLOW, REVENUE, PROFIT, ASSET, DIVIDEND, FX_RISK, ENERGY_COST, SEGMENT, ESG, RISK_MITIGATION` | Tetap — daftar pola regex lengkap ada di `config.py:76-102`, tidak diulang di sini karena panjang; lihat file untuk pattern persis tiap tag. (Versi dokumen sebelumnya menulis "15 tag" padahal daftarnya memuat 17.) |

### Layer 3 — Verification (`VerificationConfig`, `config.py:126-143`)

| Parameter | Nilai default | Runtime / status |
|---|---|---|
| `enable_numerical_gate` | `True` | Tetap — hanya lewat edit `config.py` (ablation manual). Dicek di `layer3_verification.py:186`. |
| `enable_adversarial_critique` | `True` | Tetap — sama. Dicek di `layer3_verification.py:274`. |
| `numeric_tolerance` | `0.01` (1%) | **Aktif dipakai sebagai BATAS ATAS**, bukan toleransi datar — `layer3_verification.py:206` (`cap`). Toleransi efektif per angka = `min(pembulatan angka yang ditampilkan di jawaban, 1% × max(|jawaban|, |sumber|, 1))` (`_rounding_tolerance`, `layer3_verification.py:155-164`; pencocokan di bagian `numerical_gate`). |
| `debate_rounds` | `1` | **Aktif dipakai** sebagai batas loop `for rnd in range(VERIFICATION.debate_rounds)` (`layer3_verification.py:295`). |
| `unsupported_number_penalty` | `0.10` | **Dead config — tidak dipakai di manapun.** Grep atas seluruh kode hanya menemukan deklarasinya di `config.py:140`. Penalti deterministik dinonaktifkan secara eksplisit oleh komentar di `layer3_verification.py:221-223` ("Penalti deterministik DINONAKTIFKAN (permintaan)"); `NumericalGateResult` tidak punya field `penalty`. |
| `min_confidence` | `0.70` | **Dead config — tidak dipakai di manapun.** Hanya dideklarasikan di `config.py:143` (ambang yang benar-benar dipakai adalah `SOCIOTECHNICAL.confidence_threshold_by_risk`, lihat tabel Layer 4). |

### Layer 4 — Sociotechnical (`SociotechnicalConfig`, `config.py:150-175`)

| Parameter | Nilai default | Keterangan |
|---|---|---|
| `enable_human_trigger` | `True` | Tetap. Kalau `False`, `human_judgement_trigger()` selalu `auto_accept` (`human_review.py:117-123`). |
| `confidence_threshold_by_risk` | `{"low": 0.55, "medium": 0.75, "high": 0.90}` | Ambang confidence per level risiko — dipakai `human_judgement_trigger()` (`human_review.py:115`). |
| `high_risk_tags` | `["CURRENCY", "PERCENTAGE", "DIVIDEND", "PROFIT"]` | Tag Layer 2 yang otomatis menaikkan risk ke `"high"` (`human_review.py:55-56`). |
| `medium_risk_tags` | `["CASHFLOW", "FX_RISK", "RISK_MITIGATION", "ASSET"]` | Tag yang menaikkan risk ke minimal `"medium"` (`human_review.py:57-59`). |
| `always_escalate_on_unsupported_number` | `True` | Kalau ada angka tak terdukung di gate numerik → langsung `action="escalate"`, terlepas dari confidence (`human_review.py:136-138`). |

### Eksperimen (`config.py:179-223`)

| Parameter | Nilai aktual |
|---|---|
| `EVAL_QUESTIONS` | **10 soal** (`Q1`–`Q10`), kini masing-masing punya field `risk` (`config.py:188-219`). Label **USULAN** (komentar `config.py:179-187`), belum disamakan dengan Tabel 5 tesis: Q1 medium · Q2 medium · Q3 **high** · Q4 low · Q5 low · Q6 low · Q7 **high** · Q8 medium · Q9 medium · Q10 **high**. |
| `SCENARIOS` | `["vanilla", "standard_rag", "verified"]` (`config.py:221`) |
| `RESULTS_DIR` | `"./results"` |
| `LOG_FILE` | `"./results/run_log.csv"` |

---

## (c) Prompt lengkap tiap agent (apa adanya dari kode)

Catatan yang berlaku untuk ketiga agent: setiap prompt dikirim lewat `LLMClient.complete_json()` (`llm.py:156-174`), yang **menambahkan** baris berikut ke system prompt sebelum dikirim (`llm.py:171-172`):
```
\n\nJawab HANYA dengan JSON valid. Tanpa penjelasan, tanpa markdown, tanpa ```.
```
Model untuk Scorer & Commander: `MODEL.generator_model`. Model untuk Critic: `MODEL.critic_model`. `max_tokens`/`temperature` mengikuti default `MODEL` (10000 / 0.0) karena tidak ada override per-call.

**Alur informasi antar agent:** Scorer menerima pertanyaan, konteks, jawaban, dan laporan gate. **Critic hanya menerima** pertanyaan, konteks, dan jawaban — output Scorer **tidak** dikirim ke Critic. Commander menerima ketiganya ditambah confidence+reasoning Scorer dan isu+saran Critic.

### Scorer

System (`layer3_verification.py:235-238`):
```
Anda adalah Scorer dalam sistem verifikasi jawaban laporan keuangan.
Tugas Anda menilai apakah jawaban benar-benar didukung oleh sumber yang diberikan.
Anda TIDAK menilai gaya bahasa. Anda hanya menilai kesetiaan terhadap sumber (faithfulness)
dan kebenaran faktual.
```

User prompt template (`layer3_verification.py:299-311`, `{...}` diisi runtime):
```
PERTANYAAN:
{question}

SUMBER:
{context}

JAWABAN YANG DINILAI:
{answer}

HASIL PEMERIKSAAN NUMERIK OTOMATIS:
{gate_report}

Nilai jawaban. Format: {"confidence":0.0-1.0,"supported_claims":["..."],"unsupported_claims":["..."],"reasoning":"ringkas, maksimal 2 kalimat"}
```
`fallback` jika parse JSON gagal: `{"confidence": 0.5, "unsupported_claims": [], "reasoning": ""}`.

### Critic

System (`layer3_verification.py:240-246`):
```
Anda adalah Critic yang berperan sebagai devil's advocate.
Tugas Anda MENYERANG jawaban, bukan membelanya. Asumsikan jawaban itu salah
sampai terbukti benar. Cari: angka yang tidak ada di sumber, klaim yang
melampaui isi dokumen, instruksi pertanyaan yang tidak dijawab, dan
kesimpulan yang ditarik tanpa dasar.
Jika setelah diperiksa jawaban memang solid, katakan demikian secara singkat.
Jangan mengarang cacat yang tidak ada.
```

User prompt template (`layer3_verification.py:324-334`):
```
Question: 
{question}

Context: 
{context}

Answer: 
{answer}

Format: {"issues":["..."],"severity":"none|low|medium|high","suggested_fix":"..."}
```
Skema JSON sudah lengkap (versi sebelumnya terpotong dan tanpa `suggested_fix`; diperbaiki di `d2e8417`). `fallback` jika parse gagal: `{"issues": [], "severity": "none", "suggested_fix": ""}`.

### Commander

System (`layer3_verification.py:248-251`):
```
Anda adalah Commander. Anda menerima jawaban awal, penilaian Scorer,
dan serangan Critic. Tugas Anda membuat keputusan final: pertahankan, revisi, atau
tandai sebagai tidak dapat dijawab dari sumber yang tersedia.
Prinsip: lebih baik mengatakan "tidak ditemukan dalam dokumen" daripada menebak.
```

User prompt template (`layer3_verification.py:352-365`):
```
Question:
{question}

Context:
{context}

Answer:
{answer}

Confidence={scorer_conf}; {scored.get('reasoning','')}

Critic ({severity}): {round_issues}
Fix: {critique.get('suggested_fix','')}

Format: {"final_answer":"...","confidence":0.0-1.0,"verdict":"kept|revised|not_answerable"}
```
`fallback`: `{"final_answer": answer, "confidence": scorer_conf, "verdict": "kept"}`.

### Konstruksi `gate_report` (dipakai di prompt Scorer)
`layer3_verification.py:287-294`:
- Kalau semua angka jawaban terverifikasi: `"Semua angka dalam jawaban terverifikasi di sumber."`
- Kalau ada yang tidak: `f"PERINGATAN GATE NUMERIK: {len(gate.unsupported)} angka TIDAK ditemukan di sumber manapun: {listed}"` — `listed` maksimal 8 angka pertama yang tak terdukung.

### System prompt generasi jawaban (bukan bagian DEBATE, tapi dipakai di semua skenario kecuali vanilla)
`pipeline.py:28-41`, dipakai `StandardRAGPipeline` dan `VerifiedPipeline`:
```
Anda adalah asisten analisis laporan keuangan tahunan perusahaan
terbuka Indonesia. Anda menjawab dalam Bahasa Indonesia, ringkas dan presisi.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan blok <SUMBER> yang diberikan.
2. Setiap angka dan klaim faktual WAJIB diikuti sitasi [chunk_id] sesuai
   atribut id pada blok sumbernya.
3. Jika informasi tidak ada di sumber, tulis persis:
   "Tidak ditemukan dalam dokumen yang tersedia."
   JANGAN menebak, JANGAN melengkapi dari pengetahuan umum.
4. Jangan mengubah, membulatkan, atau menghitung ulang angka yang ada
   di sumber kecuali diminta secara eksplisit.
```
`VanillaPipeline` memakai versi tanpa "ATURAN WAJIB" (`BASE_SYSTEM`, `pipeline.py:28-29`).

---

## (d) Alur data: PDF masuk → jawaban keluar

Alur berikut mengikuti `VerifiedPipeline.run()` (`pipeline.py:210-290`), skenario paling lengkap. Perbedaan `standard_rag`/`vanilla` dicatat di tiap langkah yang relevan.

1. **Ingest (sekali per dokumen, bukan per pertanyaan).** `RetrievalLayer.build()` (`layer1_retrieval.py:172-201`): kalau index sudah ada di disk (`{index_dir}/{doc_id}_chunks.pkl`), langsung `pickle.load` + `FAISS.load_local` (hemat biaya). Kalau belum: `PyPDFLoader.load()` → `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) → tiap potongan jadi objek `Chunk` dengan `chunk_id = f"{doc_id}::c{i:04d}"` (`layer1_retrieval.py:161`) → `FAISS.from_texts()` dengan embedding Voyage (`voyage-finance-2`) → `save_local()` + `pickle.dump()` chunk metadata.
   - **Dua `doc_id` berbeda.** `RetrievalLayer.doc_id` = nama file tanpa ekstensi apa adanya, mis. `22. FS KLBF 2025` (`layer1_retrieval.py:135`) — dipakai untuk nama file index dan prefix `chunk_id`. Sedangkan kolom `doc_id` di `run_log.csv` diisi `make_doc_id()` (`sources.py:36-52`), mis. `22._FS_KLBF_2025` (`runner.py` menimpa `res.doc_id`). `evaluation.py` menjembatani keduanya dengan mengambil nama index dari prefix `chunk_id` (`evaluation.py:273`). Untuk join antar skenario pakai kolom `ticker` (`evaluation.py:33,37`), bukan `doc_id`.
   - *(opsional)* `KnowledgeGraph.extract_from_chunks()` (`layer1_retrieval.py:271-299`) — 1 panggilan LLM per chunk (dibatasi `max_chunks=15`, jadi hanya 15 chunk pertama), pakai `MODEL.utility_model`, ekstrak triple `(subject, relation, object)` via NetworkX graph in-memory. Default **mati** di `runner.py` (`--enable-kg`, `runner.py:226`); di Streamlit lewat checkbox (`app.py:151,179-186`). Token ekstraksi dicatat terpisah di `results/build_costs.csv` (`runner.py:65-76,142`), tidak lagi hilang saat ledger di-reset.

2. **Layer 1 — retrieval per pertanyaan.** `retriever.search(question, top_k=RETRIEVAL.top_k_prefetch=20)` (`pipeline.py:225`) → FAISS similarity search, hasil di-skor `1/(1+L2_distance)`.
   - `standard_rag`: `top_k=8` langsung, tanpa Layer 2 (`pipeline.py:154`).
   - `vanilla`: langkah ini dilewati total — tidak ada dokumen sama sekali.

3. **(Kondisional) Knowledge Graph context.** Kalau `kg` aktif dan query mengandung tag `SEGMENT`, tarik `kg.neighbors_text(self.retriever.doc_id, hops=2)` (`pipeline.py:227-232`) untuk ditempel ke context. `neighbors_text()` mengembalikan string kosong kalau entitas tidak ada sebagai node (`layer1_retrieval.py:305-306`); karena argumen yang diberikan adalah nama file dokumen dan bukan nama entitas hasil ekstraksi, hasilnya praktis kosong kecuali ekstraktor kebetulan membuat node dengan nama itu.

4. **Layer 2 — context management.** `build_context()` (`layer2_context.py:174-213`), urutan: `tag_chunks()` (regex per chunk) → `tag_query()` (regex + intent-map per pertanyaan) → `rerank()` (`0.7×similarity + 0.3×tag_overlap`) → `apply_budget()` (ambil skor tertinggi sampai `max_chunks=6` ATAU `max_context_tokens=3000` habis) → `reorder_against_lost_in_the_middle()` (susun `[terbaik, ke-3, ke-5, ..., ke-6, ke-4, ke-2]`). Hasil dirender jadi blok `<SUMBER id="..." halaman="..." tag="...">...</SUMBER>`, ditambah `<GRAF_ENTITAS>` kalau ada, ditambah riwayat percakapan (`ConversationBuffer.render()`, tanpa limit turn, tiap giliran dipotong 150 karakter pertanyaan + 300 karakter jawaban, `layer2_context.py:149-156`) kalau `use_buffer=True`.

5. **Generasi draft jawaban.** `LLMClient.complete()` dengan `GROUNDED_SYSTEM` + prompt `"SUMBER:\n{context}\n\nPERTANYAAN:\n{question}"` (`pipeline.py:241-244`), model `MODEL.generator_model`, `temperature=0.0`.

6. **Layer 3a — numerical gate (pra-debate).** `numerical_gate(draft, managed.chunks, extra_scales=self.retriever.table_scales())` (`pipeline.py:247`). Cara kerja gate (`layer3_verification.py:166-232`):
   - Angka diekstrak dari teks dengan `extract_numbers()` (`layer3_verification.py:104-142`): sitasi `[chunk_id]` dan penomoran daftar ("1.", "2)") dibuang dulu; format Indonesia (`1.234,5`) dan Inggris (`371,335,392,068`) sama-sama dikenali; satuan (`juta`, `miliar`, `triliun`, `%`, dll.) dibaca hanya kalau menempel pada angka (kata seperti "3 kali" tidak dihitung 3k). Tiap angka membawa `value`, `is_percent`, `is_year`, `has_unit`, `decimals`, `unit_scale`.
   - Angka berbentuk tahun (1900–2099 tanpa satuan) **dikecualikan** dari penyebut gate (`layer3_verification.py:202`).
   - Skala tabel: header seperti "dalam jutaan Rupiah" / "in millions" dideteksi `detect_table_scale()` (`layer3_verification.py:61-72`). Skala dikumpulkan dari chunk yang terambil **dan** dari `extra_scales` (skala seluruh dokumen, `RetrievalLayer.table_scales()`, `layer1_retrieval.py:203-213`), karena header sering tidak ikut terambil bersama chunk badan tabelnya. Sel tabel tanpa satuan eksplisit mewarisi skala itu sebagai kandidat nilai sumber.
   - Pencocokan memakai **nilai**, bukan string, dengan toleransi `min(pembulatan yang ditampilkan, 1%)` (lihat tabel Layer 3).
   - Gate hanya memeriksa **eksistensi** nilai di konteks, bukan bahwa nilai itu milik akun yang ditanyakan (lihat bagian (g)).

7. **Layer 3b — DEBATE.** `run_debate()` (`layer3_verification.py:265-392`), sampai `debate_rounds=1` iterasi: Scorer menilai → Critic menyerang → (kalau severity bukan `none`/`low` ATAU gate gagal; `layer3_verification.py:346`) Commander memutuskan `final_answer`/`confidence`/`verdict`. Confidence akhir = `max(0, min(1, scorer_conf))` (`layer3_verification.py:381`) — **tidak dikurangi penalti gate manapun** (penalti dinonaktifkan secara sengaja, lihat tabel Layer 3).

8. **Gate numerik dijalankan ULANG** pada `debate.final_answer` (`pipeline.py:253-254`) — supaya angka baru hasil revisi Commander ikut terperiksa.

9. **Layer 4 — human judgement trigger.** `human_judgement_trigger()` (`human_review.py:97-166`): tentukan `risk_level` (dari `question_risk` — label `risk` pada `EVAL_QUESTIONS` — plus tag Layer 2, hanya naik tidak pernah turun, `human_review.py:36-60`) → bandingkan `confidence` vs `confidence_threshold_by_risk[risk]` → tiga jalur: `auto_accept` (confidence cukup & gate bersih), `review` (confidence di bawah ambang, atau risiko `high` tetapi jawaban memuat nol angka, `human_review.py:144-149`), `escalate` (ada angka tak terdukung, forced oleh `always_escalate_on_unsupported_number=True`).

10. **Buffer & output.** Kalau `use_buffer=True`, `(question, final_answer)` ditambahkan ke `ConversationBuffer` untuk pertanyaan berikutnya (`pipeline.py:264-265`). `TokenLedger.snapshot()` diambil (akumulasi seluruh panggilan LLM di langkah 5+7, per-stage). Hasil dirangkai jadi `PipelineResult` (`pipeline.py:48-92`) — dipakai `app.py` untuk render UI, dan `ResultLogger.write()` (`runner.py:56-63`) untuk append ke `results/run_log.csv`.

**Buffer sesi di tiga skenario.** Ketiga pipeline punya `ConversationBuffer` sendiri dan parameter `use_buffer`, dan memakai `render()` yang sama. Posisi riwayat di prompt sama: setelah blok `<SUMBER>`, sebelum `PERTANYAAN` (vanilla: di depan pertanyaan). Riwayat baseline **tidak** melewati Layer 2 (tanpa tagging/budget).
- Default parameter: `VanillaPipeline` dan `StandardRAGPipeline` = `False` (`pipeline.py:108,149`), `VerifiedPipeline` = `True` (`pipeline.py:216`).
- `runner.py` meneruskan `use_buffer=self.session_buffer` (default `False`, `runner.py:101,163`) ke **semua** skenario, jadi run batch default = tiap soal independen. `--session-buffer` (`runner.py:227`) menyalakannya untuk ketiga skenario; hasilnya sebaiknya ke CSV terpisah (`--out`), jangan dicampur dengan run independen.
- `app.py` **tidak** meneruskan `use_buffer` (`app.py:287-294`), jadi di Streamlit `verified` memakai riwayat (default `True`) sedangkan `standard_rag` dan `vanilla` tidak (default `False`).
- Buffer dibuat per objek pipeline, dan `ExperimentRunner` membuat pipeline baru per (dokumen, skenario) (`runner.py:146-148`), sehingga riwayat tidak bocor antar dokumen atau antar skenario.

`vanilla`: langkah 1–4 dan 6–9 dilewati total; hanya langkah 5 (generasi, tanpa `<SUMBER>`, `BASE_SYSTEM` bukan `GROUNDED_SYSTEM`) dengan `confidence` di-hardcode `0.0` (`pipeline.py:127`).
`standard_rag`: langkah 2 pakai `top_k=8` langsung (tanpa Layer 2), langkah 6 dihitung tapi **hasilnya tidak memengaruhi jawaban** (murni instrumen pengukuran, komentar eksplisit `pipeline.py:170-172`), langkah 7 dan 9 dilewati total, `confidence` di-hardcode `0.0` (`pipeline.py:178`).

---

## (e) Hardcoded / placeholder / dead config
| Item | Lokasi | Kenapa masuk daftar ini |
|---|---|---|
| `RETRIEVAL.pdf_dir` | `config.py:50-53` | Path default hardcode ke folder lokal milik satu pengembang (`C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester\Reserach Writing I\IDX30`). **Tidak dipakai di manapun** di kode — `runner.py` punya `--pdf-dir` sendiri yang independen. Field ini murni dead. |
| `scripts/link_pdfs.py` `DEFAULT_SOURCE` | `scripts/link_pdfs.py:45-49` | Path hardcode serupa (folder pribadi pengembang), dipakai sebagai default `--source` skrip ini kalau env `VERIFIED_SOURCE_PDF_DIR` tidak diset. Berfungsi (tidak dead), tapi tidak portable ke mesin lain. |
| `VERIFICATION.unsupported_number_penalty` | `config.py:140` | Dead config — dideklarasikan, tidak pernah dibaca di file manapun. Penalti numerik sudah dinonaktifkan secara eksplisit (komentar `layer3_verification.py:221-223`). |
| `VERIFICATION.min_confidence` | `config.py:143` | Dead config — dideklarasikan, tidak pernah dibaca di file manapun. Ambang confidence yang benar-benar dipakai adalah `SOCIOTECHNICAL.confidence_threshold_by_risk`. |
| Label `risk` di `EVAL_QUESTIONS` | `config.py:188-219` | Sudah diisi, tapi **usulan** dan belum disamakan dengan Tabel 5 tesis (komentar `config.py:186-187`). Label ini menentukan ambang eskalasi Layer 4, jadi mengubahnya mengubah hasil. |
| `MODEL.generator_model` / pilihan model di `app.py:100-102` | `config.py:27`, `app.py:100-102` | String model (`"claude-sonnet-4-6"`, `"claude-opus-4-8"`) di-hardcode literal, tidak divalidasi terhadap katalog model yang tersedia saat runtime — kalau ID tidak valid, error baru muncul saat SDK dipanggil (`anthropic.messages.create`). |
| `NumericalGateResult.penalty` | dihapus dari `layer3_verification.py` | Bukan lagi field di dataclass, konsisten dengan penalti yang dinonaktifkan — keputusan desain yang sudah "selesai" secara arsitektur (beda dengan dua field config di atas yang masih tersisa tak terpakai). |
| Kelas index FAISS persis (`IndexFlat`, `IndexIVF`, dst.) | tidak diset eksplisit di `layer1_retrieval.py` | **[BELUM DITENTUKAN]** — diserahkan ke default internal `langchain_community.vectorstores.FAISS.from_texts`, tidak terlihat dari kode repo ini. |
| Anotasi hallucination rate | `evaluation.py:48-64`, `runner.py:34-44,56-60` | Kolom `is_hallucination`, `hallucination_type`, `annotator_note` di `run_log.csv` **sengaja dikosongkan** oleh sistem (komentar eksplisit `runner.py:44`) — ini bukan bug, tapi placeholder yang wajib diisi manual oleh evaluator manusia sebelum `evaluation.hallucination_rate()` bisa jalan (fungsi akan `raise ValueError` kalau belum ada baris teranotasi, `evaluation.py:55`). |
| Ground truth RAGAS (`context_recall`) | `evaluation.py:373-402` | Parameter `ground_truth` di `build_ragas_dataset()` opsional dan tidak ada sumber data bawaan di repo — README (`README.md:71`) menyebut ini harus ditulis manual, belum ada di kode/data. README masih menghitung "30 × 8 = 240" jawaban acuan; dengan 10 soal jumlahnya 300. |
| Jumlah soal di README | `README.md` (tabel peta file & catatan #2) | Masih menyebut 8 soal, padahal `EVAL_QUESTIONS` berisi 10. Tidak memengaruhi eksekusi. |

---

## (f) Perubahan sejak versi dokumen sebelumnya (`6cd1156` → `59a6bd9`)

| # | File | Sebelumnya | Sekarang |
|---|------|------------|----------|
| 1 | `layer3_verification.py` | `numerical_gate` crash `KeyError: 'is_year'` pada setiap jawaban berisi angka (runner menelan error per soal, jadi baris berangka hilang diam-diam) | `extract_numbers()` mengembalikan `is_year`, `has_unit`, `decimals`, `unit_scale` |
| 2 | `layer3_verification.py` | Header "dalam jutaan Rupiah" diabaikan → hampir semua angka tabel jadi false positive → Layer 4 mengeskalasi hampir semua jawaban | Gate sadar-skala-tabel (`detect_table_scale`, `RetrievalLayer.table_scales()`, parameter `extra_scales`) |
| 3 | `layer3_verification.py` | Toleransi datar 1% | `numeric_tolerance` jadi batas atas; toleransi efektif `min(pembulatan, 1%)` |
| 4 | `layer3_verification.py` | Format angka Inggris `371,335,392,068` terbaca sebagai desimal | Dikenali sebagai satu angka |
| 5 | `layer3_verification.py` | Penomoran daftar ("1.") dihitung sebagai angka; "3 kali" terbaca 3k | `LIST_MARKER_RX` dan `(?![A-Za-z])` pada `NUMBER_RX` |
| 6 | `layer3_verification.py` | Skema JSON prompt Critic terpotong, tanpa `suggested_fix` | Skema lengkap |
| 7 | `evaluation.py` | `load_chunk_lookup` mencari nama file index yang salah → judge post-hoc menilai tanpa sumber | Nama index diambil dari prefix `chunk_id` |
| 8 | `evaluation.py` | `doc_id` vanilla (`KLBF`) ≠ runner (`22._FS_KLBF_2025`) → tidak bisa join | Kolom turunan `ticker` di `load_results()` |
| 9 | `runner.py` / `pipeline.py` | Buffer riwayat aktif default hanya di VERIFIED → jawaban Q1 bocor ke Q2..Q10, VERIFIED vs RAG tidak sebanding | Default OFF; `--session-buffer` menyalakannya |
| 10 | `runner.py` | Token ekstraksi KG hilang (ledger di-reset) | Dicatat di `results/build_costs.csv` |
| 11 | `config.py` | `EVAL_QUESTIONS` tanpa kunci `risk` → semua soal default `medium`; komentar menyebut 8 soal | Label `risk` (usulan) + komentar 10 soal |
| 12 | `pipeline.py` | Baseline tanpa buffer | `VanillaPipeline` dan `StandardRAGPipeline` punya `ConversationBuffer` dan `use_buffer` agar eksperimen sesi adil antar skenario |
| 13 | `runner.py` | `.env` tidak dibaca (hanya `app.py` yang memanggil `load_dotenv`) → `ANTHROPIC_API_KEY kosong` | `load_dotenv()` dipanggil sebelum `import config` |
| 14 | `data/docs/technical_spec.md` | Mengutip "Catatan Kritis #1–#3" dan "bagian 0" yang tidak ada di dokumen; menulis 15 tag regex | Rujukan itu dihapus (isinya kini di bagian (f) dan (g)); jumlah tag dikoreksi menjadi 17 |

---

## (g) Keterbatasan yang diketahui dan belum diselesaikan

1. **Gate hanya memeriksa eksistensi nilai.** Jawaban yang menukar "laba bruto" dengan "laba bersih" lolos selama nilainya ada di konteks (docstring `layer3_verification.py:181-184`). Harus ditangkap anotasi manusia.
2. **Gate membandingkan angka jawaban dengan chunk saja, bukan riwayat.** Pada mode sesi (`--session-buffer`), jawaban yang mengutip angka dari jawaban sebelumnya bisa terhitung "tak terdukung" padahal bukan halusinasi. Dugaan ini dari membaca kode dan belum dibuktikan dengan data; verifikasi dulu sebelum memakai angka `unsupported` dari run sesi.
3. **Knowledge Graph.** `neighbors_text(doc_id)` mencari node bernama nama file (lihat bagian (d) langkah 3), ekstraksi hanya dari 15 chunk pertama (biasanya halaman sampul/daftar isi), dan `--enable-kg` default mati.
4. **Budget Layer 2 tidak menghitung riwayat percakapan.** Riwayat ditambahkan setelah `apply_budget()`, jadi pada sesi panjang total konteks bisa melampaui `max_context_tokens`.
5. **Data lama di `results/run_log.csv`.** Berisi 300 baris vanilla (30 dokumen × 10 soal, `doc_id` berupa ticker, mis. `AADI`) dan 2 baris `verified` KLBF (Q1, Q2) yang dihasilkan dengan gate lama. Hapus 2 baris itu sebelum run penuh, karena `--resume` akan melewatinya. Skrip yang menghasilkan baris vanilla (`run_vanilla_idx30_batch.py`) tidak ada lagi di repo (hanya `.pyc` tersisa), jadi baris itu tidak bisa direproduksi dari kode yang ada.
6. **Label `risk` bersifat usulan** (lihat bagian (e)); ubah dan samakan dengan Tabel 5 tesis sebelum run resmi.
7. **Interface Streamlit tidak simetris soal buffer** (lihat bagian (d)): `verified` memakai riwayat, baseline tidak. Cukup untuk demo, tetapi bukan perbandingan yang adil.

---

*Dokumen ini statis — kalau kode berubah, perbarui bagian yang relevan alih-alih mempercayai isi dokumen ini secara membabi buta.*
