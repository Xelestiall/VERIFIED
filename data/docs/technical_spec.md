# VERIFIED — Technical Specification
Dokumen ini disusun dengan membaca kode sumber aktual di commit `6cd1156` (branch `main`), Setiap klaim disertai referensi `file:baris`. 
Bagian yang tidak bisa dipastikan dari kode ditandai **[BELUM DITENTUKAN]**.
---

## (a) Arsitektur per layer

| Layer | File implementasi | Isi |
|---    |---                |---  |
| **Layer 1 — Retrieval** | `layer1_retrieval.py` | `Chunk` (struktur data), `VoyageEmbeddings` / `LocalEmbeddings` (adapter embedding), `get_embeddings()` (factory), `RetrievalLayer` (ingest PDF → chunk → embed → FAISS, plus search), `KnowledgeGraph` (ekstraksi entitas opsional via NetworkX) |
| **Layer 2 — Context Management** | `layer2_context.py` | `tag_text`/`tag_chunks`/`tag_query` (regex tagging), `tag_overlap_score`/`rerank` (skor gabungan similarity+tag), `apply_budget` (potong sesuai token budget), `reorder_against_lost_in_the_middle` (susun ulang anti-LITM), `ConversationBuffer` (riwayat percakapan, tanpa limit turn), `build_context` (orkestrasi keempatnya → `ManagedContext`) |
| **Layer 3 — Verification** | `layer3_verification.py` | Bagian A: `extract_numbers`/`parse_id_number`/`numerical_gate` (gate numerik deterministik, non-LLM). Bagian B: `run_debate` — siklus Scorer → Critic → Commander (`SCORER_SYSTEM`, `CRITIC_SYSTEM`, `COMMANDER_SYSTEM`) |
| **Layer 4 — Sociotechnical / Human Review** | `human_review.py` | `assess_risk` (naikkan risk level berdasar tag), `build_human_prompts` (susun instruksi konkret utk analis), `human_judgement_trigger` (putuskan `auto_accept` / `review` / `escalate`) |
| **Orkestrasi + 3 skenario eksperimen** | `pipeline.py` | `VanillaPipeline` (LLM murni, tanpa dokumen), `StandardRAGPipeline` (Layer 1 saja), `VerifiedPipeline` (Layer 1+2+3+4), `PipelineResult` (skema output flat, siap jadi baris CSV), `build_pipeline()` (factory), `extract_citations()` |
| **LLM wrapper + token accounting** | `llm.py` | `LLMClient` (satu-satunya titik panggil Anthropic SDK: `.complete()`, `.complete_json()`, `.count_tokens()`), `TokenLedger` (akumulasi token per-stage + biaya), `safe_json_parse()` |
| **Konfigurasi terpusat** | `config.py` | Semua dataclass parameter (`ModelConfig`, `RetrievalConfig`, `ContextConfig`, `VerificationConfig`, `SociotechnicalConfig`), `REGEX_TAGS`, `EVAL_QUESTIONS`, API-key loader `get_api_key()` |
| **Sumber dokumen (multi-backend)** | `sources.py` | `BaseSource` + implementasi `LocalSource`, `SupabaseSource`, `GoogleDriveSource`, `SharePointSource`; `make_doc_id()` (kunci join antar CSV); `build_source()` (factory dari CLI args) |
| **Batch runner eksperimen** | `runner.py` | `ExperimentRunner` (loop dokumen × skenario × soal), `ResultLogger` (tulis `results/run_log.csv`, dukung `--resume`) |
| **Evaluasi/metrik** | `evaluation.py` | Hallucination rate (dari anotasi manual `is_hallucination`), proxy otomatis, analisis token, traceability, human-trigger analysis, "Opsi B" (`run_judge_confidence` — judge post-hoc terpisah dari confidence bawaan pipeline), integrasi RAGAS opsional |
| **UI interaktif** | `app.py` | Streamlit — upload PDF, pilih skenario, jalankan pipeline, tampilkan jawaban + confidence + expander debate live + tab Sumber/Konteks/Biaya |
| **Persiapan data sumber** | `scripts/link_pdfs.py` | Dedup PDF via SHA-256, hardlink/symlink/copy ke `./data/pdf`, exclude file non-laporan |

---

## (b) Tabel parameter aktual

Nilai diambil dari `config.py` (sumber tunggal default) dengan catatan di kolom "Runtime" bila nilainya bisa berubah saat dijalankan (Streamlit sidebar / CLI flag). Kalau tidak dicatat, nilai **tetap** — tidak ada jalur di kode yang mengubahnya.

### Model (`ModelConfig`, `config.py:24-42`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `generator_model` | `"claude-sonnet-4-6"` | Bisa diganti di sidebar `app.py:100-102` — pilihan: `"claude-sonnet-4-6"` atau `"claude-opus-4-8"`. Nilai literal ini, apa adanya dari kode; belum diverifikasi terhadap katalog model Anthropic yang berlaku saat dokumen ini ditulis. |
| `critic_model` | `"claude-sonnet-4-6"` | Di `app.py:103` **dipaksa sama dengan** `generator_model` setiap render sidebar (`MODEL.critic_model = MODEL.generator_model`) — di jalur Streamlit, ketiga model role selalu identik. Di `runner.py` (batch, tanpa Streamlit) nilai ini independen dari config, tapi kebetulan default-nya sama. |
| `utility_model` | `"claude-sonnet-4-6"` | Sama seperti `critic_model` — dipaksa sama dengan `generator_model` di `app.py:104`. Dipakai untuk ekstraksi Knowledge Graph (`layer1_retrieval.py:287`). |
| `max_tokens` | `10000` | Tidak pernah di-override per-call di kode manapun (`llm.py:120`) — berlaku untuk semua panggilan (generation, scorer, critic, commander, KG extraction). |
| `temperature` | `0.0` | Tidak pernah di-override per-call di kode manapun (`llm.py:121`) — grep atas seluruh repo tidak menemukan pemakaian `temperature=` lain selain di `llm.py`. |
| `price_per_mtok_input` | `3.00` (USD/juta token) | Tetap. |
| `price_per_mtok_output` | `15.00` (USD/juta token) | Tetap. |

### Layer 1 — Retrieval (`RetrievalConfig`, `config.py:47-72`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `index_dir` | `"./data/indexes"` | Di-override oleh `app.py` per-sesi (folder temp OS, lihat `get_session_storage()` — `app.py:53-61`). `runner.py` pakai nilai default ini. |
| `pdf_dir` | env `VERIFIED_PDF_DIR`, fallback path lokal `C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester\Reserach Writing I\IDX30` | **Tidak dipakai di manapun** selain didefinisikan (lihat bagian (e)) — `runner.py` punya argumen `--pdf-dir` sendiri (default `"./data/pdf"`) yang tidak membaca field ini. |
| `chunk_size` | `1000` (karakter) | Tetap — dipakai `RecursiveCharacterTextSplitter` (`layer1_retrieval.py:152`). |
| `chunk_overlap` | `200` (karakter) | Tetap. |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Tetap. |
| `embedding_model` | `"voyage-finance-2"` | Tetap (default `VoyageEmbeddings`). |
| `embedding_batch_size` | `64` | Tetap. |
| `top_k` | `8` | Dipakai `StandardRAGPipeline` (`pipeline.py:143`, lewat `RetrievalLayer.search()` default) dan sebagai default umum `RetrievalLayer.search()` (`layer1_retrieval.py:214`). |
| `top_k_prefetch` | `20` | Dipakai HANYA oleh `VerifiedPipeline` (`pipeline.py:208`) untuk prefetch sebelum Layer 2 memangkas ke `max_chunks`. |
| **Embedding provider** | `"voyage"` | Default parameter fungsi (`layer1_retrieval.py:110`, `app.py:173` eksplisit `"voyage"`); alternatif `"local"` (`SentenceTransformer` `intfloat/multilingual-e5-base`) hanya lewat `runner.py --embedding local` (`runner.py:196`) — tidak ada opsi ganti provider di UI Streamlit. |
| **Vector store** | FAISS lokal (`langchain_community.vectorstores.FAISS`, via `.from_texts` / `.save_local` / `.load_local`) | Jarak diukur L2 (dinyatakan eksplisit di docstring `layer1_retrieval.py:206-209`), dikonversi ke similarity via `1/(1+dist)`. Kelas index FAISS persis yang dipilih `langchain_community` di baliknya: **[BELUM DITENTUKAN]** (tidak diset eksplisit di repo ini). |

### Layer 2 — Context Management (`ContextConfig` + `REGEX_TAGS`, `config.py:76-119`)

| Parameter | Nilai default | Runtime |
|---|---|---|
| `max_context_tokens` | `3000` | Bisa diubah slider `app.py:123-129` (1000–10000, step 250) — hanya berlaku saat skenario `verified` dipilih (slider `disabled` untuk skenario lain, tapi field global tetap berubah). |
| `max_chunks` | `6` | Tetap — tidak ada kontrol UI. |
| `weight_similarity` | `0.7` | Bisa diubah slider `app.py:112-120` (0.0–1.0, step 0.05). |
| `weight_tag_match` | `0.3` | Turunan otomatis: `1 - weight_similarity` (dibulatkan 2 desimal, `app.py:121`), bukan slider independen. |
| `reorder_lost_in_the_middle` | `True` | Tetap — tidak ada kontrol UI, hanya lewat edit `config.py` langsung (jalur ablation manual). |
| `chars_per_token` | `3.5` | Tetap — dipakai estimasi cepat token (`layer2_context.py:100`, fallback `llm.py:186`) saat `count_tokens()` API gagal. |
| `REGEX_TAGS` | 15 tag: `CURRENCY, NUMBER, PERCENTAGE, YEAR, NOTE_REF, TABLE_HINT, AUDITED, CASHFLOW, REVENUE, PROFIT, ASSET, DIVIDEND, FX_RISK, ENERGY_COST, SEGMENT, ESG, RISK_MITIGATION` | Tetap — daftar pola regex lengkap ada di `config.py:76-101`, tidak diulang di sini karena panjang; lihat file untuk pattern persis tiap tag. |

### Layer 3 — Verification (`VerificationConfig`, `config.py:126-143`)

| Parameter | Nilai default | Runtime / status |
|---|---|---|
| `enable_numerical_gate` | `True` | Tetap — hanya lewat edit `config.py` (ablation manual). |
| `enable_adversarial_critique` | `True` | Tetap — sama. |
| `numeric_tolerance` | `0.01` (1%) | **Aktif dipakai** — `layer3_verification.py:108,115`, toleransi relatif saat mencocokkan nilai angka jawaban vs sumber. |
| `debate_rounds` | `1` | **Aktif dipakai** sebagai batas loop `for rnd in range(VERIFICATION.debate_rounds)` (`layer3_verification.py:197`). |
| `unsupported_number_penalty` | `0.10` | **Dead config — tidak dipakai di manapun.** Grep atas seluruh repo hanya menemukan deklarasinya di `config.py:140`. Penalti deterministik sudah dinonaktifkan secara eksplisit oleh komentar di `layer3_verification.py:123-125` ("Penalti deterministik DINONAKTIFKAN (permintaan)"); `NumericalGateResult` bahkan sudah tidak punya field `penalty` lagi. |
| `min_confidence` | `0.70` | **Dead config — tidak dipakai di manapun.** Hanya dideklarasikan di `config.py:143`; tidak direferensikan file lain manapun (ambang yang benar-benar dipakai adalah `SOCIOTECHNICAL.confidence_threshold_by_risk`, lihat tabel Layer 4). |

### Layer 4 — Sociotechnical (`SociotechnicalConfig`, `config.py:150-175`)

| Parameter | Nilai default | Keterangan |
|---|---|---|
| `enable_human_trigger` | `True` | Tetap. |
| `confidence_threshold_by_risk` | `{"low": 0.55, "medium": 0.75, "high": 0.90}` | Ambang confidence per level risiko — dipakai `human_judgement_trigger()` (`human_review.py:115`). |
| `high_risk_tags` | `["CURRENCY", "PERCENTAGE", "DIVIDEND", "PROFIT"]` | Tag Layer 2 yang otomatis menaikkan risk ke `"high"`. |
| `medium_risk_tags` | `["CASHFLOW", "FX_RISK", "RISK_MITIGATION", "ASSET"]` | Tag yang menaikkan risk ke minimal `"medium"`. |
| `always_escalate_on_unsupported_number` | `True` | Kalau ada angka tak terdukung di gate numerik → langsung `action="escalate"`, terlepas dari confidence. |

### Eksperimen (`config.py:180-207`)

| Parameter | Nilai aktual |
|---|---|
| `EVAL_QUESTIONS` | **10 soal** (`Q1`–`Q10`) — komentar di `config.py:181` menyebut "8 pertanyaan dari Tabel 3.2 tesis", tapi array yang benar-benar dieksekusi berisi 10 entri. Tidak satupun entri punya field `risk` (lihat Catatan Kritis #3). |
| `SCENARIOS` | `["vanilla", "standard_rag", "verified"]` |
| `RESULTS_DIR` | `"./results"` |
| `LOG_FILE` | `"./results/run_log.csv"` |

---

## (c) Prompt lengkap tiap agent (apa adanya dari kode)

Catatan yang berlaku untuk ketiga agent: setiap prompt dikirim lewat `LLMClient.complete_json()` (`llm.py:156-174`), yang **menambahkan** baris berikut ke system prompt sebelum dikirim:
```
\n\nJawab HANYA dengan JSON valid. Tanpa penjelasan, tanpa markdown, tanpa ```.
```
Model untuk Scorer & Commander: `MODEL.generator_model`. Model untuk Critic: `MODEL.critic_model`. `max_tokens`/`temperature` mengikuti default `MODEL` (10000 / 0.0) karena tidak ada override per-call.

### Scorer

System (`layer3_verification.py:137-140`):
```
Anda adalah Scorer dalam sistem verifikasi jawaban laporan keuangan.
Tugas Anda menilai apakah jawaban benar-benar didukung oleh sumber yang diberikan.
Anda TIDAK menilai gaya bahasa. Anda hanya menilai kesetiaan terhadap sumber (faithfulness)
dan kebenaran faktual.
```

User prompt template (`layer3_verification.py:202-208`, `{...}` diisi runtime):
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

System (`layer3_verification.py:142-148`):
```
Anda adalah Critic yang berperan sebagai devil's advocate.
Tugas Anda MENYERANG jawaban, bukan membelanya. Asumsikan jawaban itu salah
sampai terbukti benar. Cari: angka yang tidak ada di sumber, klaim yang
melampaui isi dokumen, instruksi pertanyaan yang tidak dijawab, dan
kesimpulan yang ditarik tanpa dasar.
Jika setelah diperiksa jawaban memang solid, katakan demikian secara singkat.
Jangan mengarang cacat yang tidak ada.
```

User prompt template — **disalin literal apa adanya, termasuk cacatnya** (`layer3_verification.py:226-230`):
```
Question: 
{question}

Context: 
{context}

Answer: 
{answer}

{"issues":["..."],"severity":"none|low|medium|high",
```
Ini bukan salah salin — kode sumbernya memang begitu (dua string literal Python yang bersebelahan otomatis digabung: `'{"issues":["..."],' '"severity":"none|low|medium|high",'`). Skema JSON yang diminta **terpotong**: tidak ada penutup `}`, tidak menyebut field `suggested_fix` sama sekali (padahal dibaca balik di `layer3_verification.py:243,259,278`), dan tidak ada lagi instruksi eksplisit "Serang jawaban ini" yang ada di versi sebelumnya. `fallback` jika parse gagal: `{"issues": [], "severity": "none", "suggested_fix": ""}`.

### Commander

System (`layer3_verification.py:150-153`):
```
Anda adalah Commander. Anda menerima jawaban awal, penilaian Scorer,
dan serangan Critic. Tugas Anda membuat keputusan final: pertahankan, revisi, atau
tandai sebagai tidak dapat dijawab dari sumber yang tersedia.
Prinsip: lebih baik mengatakan "tidak ditemukan dalam dokumen" daripada menebak.
```

User prompt template (`layer3_verification.py:254-261`):
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
`layer3_verification.py:189-195`:
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

Alur berikut mengikuti `VerifiedPipeline.run()` (`pipeline.py:193-269`), skenario paling lengkap. Perbedaan `standard_rag`/`vanilla` dicatat di tiap langkah yang relevan.

1. **Ingest (sekali per dokumen, bukan per pertanyaan).** `RetrievalLayer.build()` (`layer1_retrieval.py:172-201`): kalau index sudah ada di disk (`{index_dir}/{doc_id}_chunks.pkl`), langsung `pickle.load` + `FAISS.load_local` (hemat biaya). Kalau belum: `PyPDFLoader.load()` → `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) → tiap potongan jadi objek `Chunk` dengan `chunk_id = f"{doc_id}::c{i:04d}"` → `FAISS.from_texts()` dengan embedding Voyage (`voyage-finance-2`) → `save_local()` + `pickle.dump()` chunk metadata.
   - *(opsional)* `KnowledgeGraph.extract_from_chunks()` (`layer1_retrieval.py:263-291`) — 1 panggilan LLM per chunk (dibatasi `max_chunks=15`), pakai `MODEL.utility_model`, ekstrak triple `(subject, relation, object)` via NetworkX graph in-memory.

2. **Layer 1 — retrieval per pertanyaan.** `retriever.search(question, top_k=RETRIEVAL.top_k_prefetch=20)` (`pipeline.py:208`) → FAISS similarity search, hasil di-skor `1/(1+L2_distance)`.
   - `standard_rag`: `top_k=8` langsung, tanpa Layer 2 (`pipeline.py:143`).
   - `vanilla`: langkah ini dilewati total — tidak ada dokumen sama sekali.

3. **(Kondisional) Knowledge Graph context.** Kalau `kg` aktif dan query mengandung tag `SEGMENT`, tarik `kg.neighbors_text(doc_id, hops=2)` (`pipeline.py:210-214`) untuk ditempel ke context.

4. **Layer 2 — context management.** `build_context()` (`layer2_context.py:174-213`), urutan: `tag_chunks()` (regex per chunk) → `tag_query()` (regex + intent-map per pertanyaan) → `rerank()` (`0.7×similarity + 0.3×tag_overlap`) → `apply_budget()` (ambil skor tertinggi sampai `max_chunks=6` ATAU `max_context_tokens=3000` habis) → `reorder_against_lost_in_the_middle()` (susun `[terbaik, ke-3, ke-5, ..., ke-6, ke-4, ke-2]`). Hasil dirender jadi blok `<SUMBER id="..." halaman="..." tag="...">...</SUMBER>`, ditambah `<GRAF_ENTITAS>` kalau ada, ditambah riwayat percakapan (`ConversationBuffer`, tanpa limit turn) kalau `use_buffer=True`.

5. **Generasi draft jawaban.** `LLMClient.complete()` dengan `GROUNDED_SYSTEM` + prompt `"SUMBER:\n{context}\n\nPERTANYAAN:\n{question}"` (`pipeline.py:224-227`), model `MODEL.generator_model`, `temperature=0.0`.

6. **Layer 3a — numerical gate (pra-debate).** `numerical_gate(draft, managed.chunks)` (`pipeline.py:230`) — **lihat Catatan Kritis #1: baris ini akan crash (`KeyError: 'is_year'`) kalau draft mengandung angka apapun.**

7. **Layer 3b — DEBATE.** `run_debate()` (`layer3_verification.py:167-292`), sampai `debate_rounds=1` iterasi: Scorer menilai → Critic menyerang → (kalau severity bukan `none`/`low` ATAU gate gagal) Commander memutuskan `final_answer`/`confidence`/`verdict`. Confidence akhir = `max(0, min(1, scorer_conf))` — **tidak lagi dikurangi penalti gate manapun** (penalti sudah dinonaktifkan secara sengaja, lihat tabel Layer 3).

8. **Gate numerik dijalankan ULANG** pada `debate.final_answer` (`pipeline.py:236`) — supaya angka baru hasil revisi Commander ikut terperiksa. (Sama-sama berisiko crash seperti langkah 6.)

9. **Layer 4 — human judgement trigger.** `human_judgement_trigger()` (`human_review.py:97-166`): tentukan `risk_level` (dari `question_risk` config + tag Layer 2, hanya naik tidak pernah turun) → bandingkan `confidence` vs `confidence_threshold_by_risk[risk]` → tiga jalur: `auto_accept` (confidence cukup & gate bersih), `review` (confidence kurang, atau butuh angka tapi jawaban nol angka), `escalate` (ada angka tak terdukung, forced oleh `always_escalate_on_unsupported_number=True`).

10. **Buffer & output.** Kalau `use_buffer=True`, `(question, final_answer)` ditambahkan ke `ConversationBuffer` untuk pertanyaan berikutnya dalam sesi yang sama. `TokenLedger.snapshot()` diambil (akumulasi seluruh panggilan LLM di langkah 5+7, per-stage). Hasil dirangkai jadi `PipelineResult` (`pipeline.py:47-89`) — dipakai `app.py` untuk render UI, dan `ResultLogger.write()` (`runner.py:43-58`) untuk append ke `results/run_log.csv`.

`vanilla`: langkah 1–4 dan 6–9 dilewati total; hanya langkah 5 (generasi, tanpa `<SUMBER>`, `BASE_SYSTEM` bukan `GROUNDED_SYSTEM`) dengan `confidence` di-hardcode `0.0` (`pipeline.py:119`).
`standard_rag`: langkah 2 pakai `top_k=8` langsung (tanpa Layer 2), langkah 6 dihitung tapi **hasilnya tidak memengaruhi jawaban** (murni instrumen pengukuran, komentar eksplisit `pipeline.py:153-155`), langkah 7 dan 9 dilewati total, `confidence` di-hardcode `0.0` (`pipeline.py:161`).

---

## (e) Hardcoded / placeholder / dead config
| Item | Lokasi | Kenapa masuk daftar ini |
|---|---|---|
| `RETRIEVAL.pdf_dir` | `config.py:50-53` | Path default hardcode ke folder lokal milik satu pengembang (`C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester\Reserach Writing I\IDX30`). **Tidak dipakai di manapun** di kode — `runner.py` punya `--pdf-dir` sendiri yang independen. Field ini murni dead. |
| `scripts/link_pdfs.py` `DEFAULT_SOURCE` | `scripts/link_pdfs.py:45-49` | Path hardcode serupa (folder pribadi pengembang), dipakai sebagai default `--source` skrip ini kalau env `VERIFIED_SOURCE_PDF_DIR` tidak diset. Berfungsi (tidak dead), tapi tidak portable ke mesin lain. |
| `VERIFICATION.unsupported_number_penalty` | `config.py:140` | Dead config — dideklarasikan, tidak pernah dibaca di file manapun. Penalti numerik sudah dinonaktifkan secara eksplisit (komentar `layer3_verification.py:123-125`). |
| `VERIFICATION.min_confidence` | `config.py:143` | Dead config — dideklarasikan, tidak pernah dibaca di file manapun. Ambang confidence yang benar-benar dipakai adalah `SOCIOTECHNICAL.confidence_threshold_by_risk`. |
| `EVAL_QUESTIONS[*]["risk"]` | `config.py:182-203` | Field `risk` tidak ada sama sekali di 10 entri soal — desain Layer 4 mengasumsikan ini diisi manual per soal (docstring `human_review.py:44`), tapi implementasinya belum. Efektif: semua soal mulai dari `risk_level="medium"` kecuali dinaikkan otomatis oleh tag. |
| Komentar jumlah soal | `config.py:181` | Komentar bilang "8 pertanyaan dari Tabel 3.2 tesis", array aktual berisi 10 (`Q1`–`Q10`). Tidak memengaruhi eksekusi, tapi dokumentasi-vs-kode tidak sinkron. |
| `MODEL.generator_model` / pilihan model di `app.py:100-102` | `config.py:27`, `app.py:100-102` | String model (`"claude-sonnet-4-6"`, `"claude-opus-4-8"`) di-hardcode literal, tidak divalidasi terhadap katalog model yang tersedia saat runtime — kalau ID tidak valid, error baru muncul saat SDK dipanggil (`anthropic.messages.create`). |
| `NumericalGateResult.penalty` | dihapus dari `layer3_verification.py` (lihat diff komit) | Bukan lagi field di dataclass, konsisten dengan penalti yang dinonaktifkan — dicatat di sini supaya jelas ini bukan bug, tapi keputusan desain yang sudah "selesai" secara arsitektur (beda dengan dua item `unsupported_number_penalty`/`min_confidence` di atas yang masih nyisa sebagai field config yang tak terpakai). |
| `Critic` JSON-schema di prompt | `layer3_verification.py:230` | Bukan hardcoded value, tapi string prompt yang secara struktural rusak (lihat Catatan Kritis #2 dan bagian (c)) — berisiko menurunkan kualitas/parsing output Critic karena field `suggested_fix` tidak lagi diminta eksplisit dalam skema. |
| Kelas index FAISS persis (`IndexFlat`, `IndexIVF`, dst.) | tidak diset eksplisit di `layer1_retrieval.py` | **[BELUM DITENTUKAN]** — diserahkan ke default internal `langchain_community.vectorstores.FAISS.from_texts`, tidak terlihat dari kode repo ini. |
| Anotasi hallucination rate | `evaluation.py:35-51`, `runner.py:38-39` | Kolom `is_hallucination`, `hallucination_type`, `annotator_note` di `run_log.csv` **sengaja dikosongkan** oleh sistem (komentar eksplisit `runner.py:39`) — ini bukan bug, tapi placeholder yang wajib diisi manual oleh evaluator manusia sebelum `evaluation.hallucination_rate()` bisa jalan (fungsi akan `raise ValueError` kalau belum ada baris teranotasi). |
| Ground truth RAGAS (`context_recall`) | `evaluation.py:353-366` | Parameter `ground_truth` di `build_ragas_dataset()` opsional dan tidak ada sumber data bawaan di repo — README (`README.md:71`) menyebut ini harus ditulis manual (240 jawaban acuan), belum ada di kode/data. |

---

*Dokumen ini statis — kalau kode berubah (terutama tiga bug di bagian 0), perbarui bagian yang relevan alih-alih mempercayai isi dokumen ini secara membabi buta.*
