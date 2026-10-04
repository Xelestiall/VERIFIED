# Perbaikan kode VERIFIED — 4 Okt 2026

File asli sebelum perubahan disimpan di `_backup_sebelum_perbaikan_2026-10-04/`.
Jalankan `python -m pytest tests -q` (12 tes) setelah menyalin ke environment-mu.

## Bug yang diperbaiki (terbukti dengan data KLBF di run_log.csv)

| # | File | Masalah | Dampak sebelum perbaikan |
|---|------|---------|--------------------------|
| 1 | layer3_verification.py | `KeyError: 'is_year'` di `numerical_gate` | Gate crash di SETIAP jawaban berisi angka; `runner.py` menelan error per-soal, jadi baris RAG/VERIFIED berangka hilang diam-diam |
| 2 | layer3_verification.py | Header "dalam jutaan Rupiah" tidak ditangani | 35 dari 35 angka "tak terdukung" pada 2 jawaban smoke-test ternyata salinan persis dari tabel (false positive 100%) -> Layer 4 mengeskalasi hampir semua jawaban |
| 3 | layer3_verification.py | Toleransi datar 1% | Angka halusinasi yang digeser 0,5% lolos 100%; digeser 10% masih lolos 12% (kebetulan dekat dengan angka lain) |
| 4 | layer3_verification.py | Format angka Inggris ("371,335,392,068") dibaca sebagai desimal | Sumber BUMI/AADI salah baca |
| 5 | layer3_verification.py | Enumerator daftar ("1.", "2.") dihitung sebagai angka; "3 kali" terbaca 3k | Penyebut gate kotor |
| 6 | layer3_verification.py | Skema JSON prompt Critic terpotong | Format keluaran Critic tidak terdefinisi |
| 7 | evaluation.py | `load_chunk_lookup` mencari `22._FS_KLBF_2025_chunks.pkl`, padahal file index bernama `22. FS KLBF 2025_chunks.pkl` | Judge post-hoc menilai semua jawaban RAG/VERIFIED tanpa sumber |
| 8 | evaluation.py | `doc_id` vanilla = `KLBF`, runner = `22._FS_KLBF_2025` | Tidak bisa join/pivot antar skenario; kini ada kolom `ticker` |
| 9 | runner.py / pipeline.py | Buffer riwayat aktif default di VERIFIED saja | Jawaban Q1 bocor ke konteks Q2..Q10: soal tidak independen & VERIFIED vs RAG tidak sebanding. Kini default OFF; `--session-buffer` untuk uji sesi |
| 10 | runner.py | Token ekstraksi KG hilang (ledger di-reset) | Biaya VERIFIED+KG terlalu rendah. Kini dicatat di `results/build_costs.csv` |
| 11 | config.py | `EVAL_QUESTIONS` tanpa kunci `risk` | Semua soal default "medium"; label risiko USULAN ditambahkan (ubah bila perlu) |

## Perubahan perilaku yang harus kamu catat di tesis (Tabel 7)

- `numeric_tolerance = 0.01` sekarang berfungsi sebagai **batas atas**; toleransi efektif =
  `min(pembulatan angka yang ditampilkan di jawaban, 1%)`.
- Gate sadar-skala tabel: skala dikumpulkan dari chunk terambil + seluruh dokumen
  (`RetrievalLayer.table_scales()`).

## Belum diperbaiki (butuh keputusanmu — lihat laporan)

- Knowledge Graph: `neighbors_text(doc_id)` mencari node bernama nama-file, sehingga selalu kosong;
  ekstraksi hanya dari 15 chunk pertama (halaman sampul/daftar isi); `--enable-kg` default mati.
- Gate hanya memeriksa **eksistensi** nilai di konteks, bukan bahwa nilai itu milik akun yang ditanya.
- 2 baris `verified` KLBF di run_log.csv memakai gate lama; hapus sebelum run penuh (karena `--resume`
  akan melewatinya).
- Baris vanilla (300) dihasilkan skrip `run_vanilla_idx30_batch.py` yang sudah tidak ada di repo
  (hanya .pyc tersisa) dan memakai `doc_id` ticker.
