"""
link_pdfs.py — Siapkan ./data/pdf dari folder sumber, sambil menyaring
file yang tidak boleh ikut eksperimen.

KENAPA PERLU SKRIP, BUKAN COPY-PASTE MANUAL:

Folder sumber IDX30 mengandung dua jenis file yang kalau ikut ter-proses
akan merusak hasil TANPA memunculkan error apa pun:

  1. DUPLIKAT SALAH NAMA. "20, FS INKP 2025.pdf" (pakai koma) ternyata
     byte-identical dengan "19. FS INDF 2025.pdf". Kalau ikut diproses,
     baris ber-doc_id INKP di run_log.csv sebenarnya berisi laporan
     keuangan INDF. Kesalahan seperti ini tidak akan ketahuan dari
     membaca CSV hasil — hanya ketahuan dari membandingkan sha256.
  2. FILE NON-LAPORAN. "Evaluasi Indeks ....pdf" ikut ter-glob oleh
     pola "*.pdf" padahal itu pengumuman bursa, bukan laporan keuangan.

Deteksi duplikat dilakukan lewat sha256, bukan lewat daftar nama file
yang di-hardcode, supaya kalau nanti ada file lain yang ke-copy dua kali
skrip ini tetap menangkapnya.

File sumber TIDAK diubah atau dihapus. Yang dibuat di ./data/pdf adalah
hardlink (kalau satu volume), symlink, atau — terpaksa — salinan.

Pakai:
    python scripts/link_pdfs.py
    python scripts/link_pdfs.py --source "D:\\folder\\lain" --dest ./data/pdf
    python scripts/link_pdfs.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

# Biar bisa `import sources` waktu skrip dijalankan dari root repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources import make_doc_id  # noqa: E402


DEFAULT_SOURCE = os.getenv(
    "VERIFIED_SOURCE_PDF_DIR",
    r"C:\Users\axelc\OneDrive - Bina Nusantara\_BINUS\8th Semester"
    r"\Reserach Writing I\IDX30",
)

# Substring (lowercase) untuk file yang jelas bukan laporan keuangan emiten.
EXCLUDE_SUBSTRINGS = ("evaluasi indeks",)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def link_one(src: str, dst: str) -> str:
    """
    Hubungkan src -> dst semurah mungkin. Kembalikan metode yang dipakai.

    Hardlink didahulukan: nol byte tambahan, dan tidak butuh hak admin
    (beda dengan symlink di Windows yang minta Developer Mode). Syaratnya
    src dan dst harus di volume yang sama.
    """
    for method, fn in (("hardlink", os.link), ("symlink", os.symlink)):
        try:
            fn(src, dst)
            return method
        except (OSError, NotImplementedError, AttributeError):
            continue
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    p = argparse.ArgumentParser(description="Siapkan ./data/pdf dari folder sumber")
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--dest", default="./data/pdf")
    p.add_argument("--dry-run", action="store_true",
                   help="tampilkan rencana tanpa menulis apa pun")
    args = p.parse_args()

    if not os.path.isdir(args.source):
        print(f"ERROR: folder sumber tidak ada: {args.source}", file=sys.stderr)
        print("       Set VERIFIED_SOURCE_PDF_DIR atau pakai --source", file=sys.stderr)
        return 1

    names = sorted(n for n in os.listdir(args.source) if n.lower().endswith(".pdf"))
    if not names:
        print(f"ERROR: tidak ada PDF di {args.source}", file=sys.stderr)
        return 1

    print(f"Sumber : {args.source}")
    print(f"Tujuan : {os.path.abspath(args.dest)}")
    print(f"Ditemukan {len(names)} file PDF\n")

    seen: dict[str, str] = {}      # sha256 -> nama file pertama yang memakainya
    accepted: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for name in names:
        src = os.path.join(args.source, name)

        low = name.lower()
        if any(s in low for s in EXCLUDE_SUBSTRINGS):
            skipped.append((name, "bukan laporan keuangan emiten"))
            continue

        digest = sha256_file(src)
        if digest in seen:
            skipped.append((name, f"DUPLIKAT byte-identical dengan '{seen[digest]}'"))
            continue

        seen[digest] = name
        accepted.append((name, digest))

    for name, reason in skipped:
        print(f"  SKIP   {name}\n         -> {reason}")
    if skipped:
        print()

    if args.dry_run:
        print(f"[dry-run] {len(accepted)} file akan dihubungkan, "
              f"{len(skipped)} dilewati. Tidak ada yang ditulis.")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    methods: dict[str, int] = {}
    for name, _ in accepted:
        src = os.path.join(args.source, name)
        dst = os.path.join(args.dest, name)
        if os.path.exists(dst):
            methods["sudah ada"] = methods.get("sudah ada", 0) + 1
            continue
        method = link_one(src, dst)
        methods[method] = methods.get(method, 0) + 1

    print(f"OK: {len(accepted)} PDF siap di {os.path.abspath(args.dest)}")
    print(f"    metode: {', '.join(f'{k}={v}' for k, v in sorted(methods.items()))}")
    print(f"    dilewati: {len(skipped)}")
    print("\nContoh doc_id yang akan dipakai di manifest.csv & run_log.csv:")
    for name, _ in accepted[:3]:
        print(f"    {name:<32} -> {make_doc_id(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
