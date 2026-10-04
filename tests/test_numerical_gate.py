"""
Uji regresi untuk gate numerik Layer 3 (jalankan: python -m pytest tests -q).

Kasus-kasus ini lahir dari bug nyata pada smoke test KLBF (Okt 2026):
  * KeyError 'is_year' -> gate crash pada SETIAP jawaban berangka
  * header 'dalam jutaan Rupiah' diabaikan -> 100% angka tabel jadi false positive
  * toleransi datar 1% meloloskan angka halusinasi yang bergeser 0,5%
  * format angka Inggris ('371,335,392,068') terbaca sebagai desimal
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_retrieval import Chunk
import layer3_verification as L3


def gate(answer, source, extra_scales=None):
    return L3.numerical_gate(answer, [Chunk("d::c0001", source, page=1)], extra_scales=extra_scales)


HEADER = "LAPORAN POSISI KEUANGAN (dalam Jutaan Rupiah) Pendapatan neto 12.345.678 Laba 1.234.567"
BODY = "Pendapatan neto 12.345.678 Laba 1.234.567"  # chunk badan tabel TANPA header


def test_tidak_crash_pada_angka_dan_tahun():
    r = gate("Pada tahun 2025 pendapatan Rp12.345.678 juta", BODY, {1e6})
    assert r.passed and r.total_numbers == 1  # tahun dikecualikan


def test_skala_dari_header_chunk_yang_sama():
    assert gate("Pendapatan Rp12,3 triliun", HEADER).passed


def test_skala_dari_tingkat_dokumen_saat_chunk_tanpa_header():
    assert gate("Pendapatan Rp12,3 triliun", BODY, {1e6}).passed


def test_tanpa_info_skala_tidak_boleh_lolos():
    assert not gate("Pendapatan Rp12,3 triliun", BODY).passed


def test_angka_karangan_ditolak():
    assert not gate("Pendapatan Rp13,5 triliun", BODY, {1e6}).passed


def test_angka_presisi_bergeser_kecil_ditolak():
    # 12.407.406 vs 12.345.678 = +0,5%; toleransi datar 1% dulu meloloskannya.
    assert not gate("Pendapatan Rp12.407.406 juta", BODY, {1e6}).passed


def test_enumerator_daftar_tidak_dihitung():
    r = gate("1. Pendapatan Rp12.345.678 juta\n2. Laba Rp1.234.567 juta", BODY, {1e6})
    assert r.passed and r.total_numbers == 2


def test_satuan_tidak_menempel_pada_kata():
    assert L3.extract_numbers("naik 3 kali lipat")[0]["value"] == 3.0


def test_format_angka_inggris_pada_sumber():
    r = gate("Nilai 371.335.392.068", "Entitas (full amount) 371,335,392,068 bunga")
    assert r.passed


def test_desimal_indonesia_tetap_desimal():
    assert L3.extract_numbers("rasio 4,7")[0]["value"] == 4.7


def test_header_variasi_penulisan():
    assert gate("Rp5,2 triliun", "(dalam juta Rp) Pendapatan 5.200.000").passed
    assert gate("US$3,5 juta", "(Disajikan dalam ribuan Dolar AS) Laba 3.500").passed


def test_persen():
    assert gate("Margin 12,5%", "Margin laba 12,5% pada 2025").passed
