"""
sources.py - 
mengubah "dokumen di suatu tempat"
menjadi "file PDF di folder lokal", lalu menyerahkan path lokalnya ke
runner. Setelah itu layer1_retrieval.py jalan persis seperti sebelumnya.
 
Kenapa fetch-then-process, bukan streaming langsung dari cloud?
  1. Reproducibility. Run lo harus bisa diulang. Kalau file di bucket
     diganti orang lain di tengah eksperimen, lo nggak akan pernah tahu.
     Adapter ini mencatat sha256 tiap file ke manifest.csv -> itu bukti
     di sidang bahwa run A dan run B memakai dokumen yang sama persis.
  2. Cache index. FAISS index di-simpan per doc_id. Kalau nama file
     berubah tiap fetch, index-nya ke-rebuild terus dan lo bayar
     embedding berulang kali.
  3. Debugging. Kalau parsing PDF gagal, lo bisa buka filenya langsung.
 
CATATAN KEAMANAN: service key / client secret JANGAN pernah masuk ke
git. Semuanya dibaca dari environment variable di bawah.
"""
 
from __future__ import annotations
 
import csv
import glob
import hashlib
import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
 
 
# ==========================================================
# Model data
# ==========================================================
def make_doc_id(filename: str) -> str:
    """
    Ubah nama file jadi ID stabil untuk penamaan index FAISS dan kolom
    doc_id di manifest.csv / run_log.csv.

    SATU-SATUNYA tempat aturan ini didefinisikan. runner.py meng-import
    fungsi ini alih-alih menghitung ulang: nama file laporan mengandung
    spasi, titik, dan koma ("7, FS BBCA 2025.pdf"), jadi dua implementasi
    yang sedikit berbeda menghasilkan doc_id berbeda dan bikin manifest.csv
    tidak bisa di-join dengan run_log.csv -- padahal justru itu alasan
    modul ini ada.

    >>> make_doc_id("7, FS BBCA 2025.pdf")
    '7_FS_BBCA_2025'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


@dataclass
class RemoteDoc:
    """Satu dokumen di sisi storage, belum tentu sudah diunduh."""
 
    filename: str          # nama file final di disk, mis. "BBCA_2025.pdf"
    remote_ref: str        # path/ID di sisi storage (beda tiap provider)
    revision: str = ""     # etag / md5 / updated_at — penanda "versi"
    size: Optional[int] = None
 
    @property
    def doc_id(self) -> str:
        """
        ID stabil untuk penamaan FAISS index dan kolom di run_log.csv.
 
        Sengaja diturunkan dari NAMA FILE, bukan dari ID internal storage.
        Alasannya: kalau lo pindah dari Drive ke Supabase di tengah jalan,
        doc_id harus tetap sama supaya hasil lama masih bisa dibandingkan.
        """
        return make_doc_id(self.filename)
 
 
# ==========================================================
# Base class — semua logika cache ada di sini, sekali saja
# ==========================================================
class BaseSource:
    """
    Subclass cukup mengisi dua method: list_documents() dan _read_bytes().
    Caching, hashing, dan penulisan manifest sudah ditangani di sini.
    """
 
    name = "base"
 
    def list_documents(self) -> List[RemoteDoc]:
        raise NotImplementedError
 
    def _read_bytes(self, doc: RemoteDoc) -> bytes:
        raise NotImplementedError
 
    # ------------------------------------------------------
    def fetch(self, doc: RemoteDoc, cache_dir: str) -> str:
        """
        Unduh kalau perlu, kembalikan path lokal.
 
        Skip download kalau file sudah ada DAN revision-nya cocok dengan
        yang tercatat di file .rev. Jadi run kedua lo nggak menunggu
        download 30 PDF lagi. Kalau file di storage diperbarui,
        revision berubah -> otomatis diunduh ulang.
        """
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, doc.filename)
        rev_path = path + ".rev"
 
        if os.path.exists(path):
            cached_rev = ""
            if os.path.exists(rev_path):
                with open(rev_path, encoding="utf-8") as f:
                    cached_rev = f.read().strip()
            # revision kosong (provider tidak menyediakan) -> percaya cache
            if not doc.revision or cached_rev == doc.revision:
                return path
            print(f"  [{self.name}] {doc.filename}: versi berubah, unduh ulang")
 
        print(f"  [{self.name}] mengunduh {doc.filename} ...")
        data = self._read_bytes(doc)
        with open(path, "wb") as f:
            f.write(data)
        with open(rev_path, "w", encoding="utf-8") as f:
            f.write(doc.revision or "")
        return path
 
    def sync(
        self, cache_dir: str = "./data/pdf", manifest_path: Optional[str] = None
    ) -> List[str]:
        """
        Tarik semua dokumen, tulis manifest, kembalikan daftar path lokal.
 
        manifest.csv itu lampiran metodologi lo: berisi sha256 tiap PDF
        yang benar-benar dipakai di run ini. Kalau penguji minta bukti
        reproducibility, ini jawabannya.
        """
        docs = self.list_documents()
        if not docs:
            raise RuntimeError(
                f"[{self.name}] tidak menemukan dokumen apa pun. "
                "Cek nama bucket/folder dan izin akses."
            )
 
        print(f"[{self.name}] {len(docs)} dokumen ditemukan")
        rows, paths = [], []
        for d in docs:
            p = self.fetch(d, cache_dir)
            paths.append(p)
            rows.append(
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "source": self.name,
                    "remote_ref": d.remote_ref,
                    "revision": d.revision,
                    "sha256": sha256_file(p),
                    "size_bytes": os.path.getsize(p),
                }
            )
 
        manifest_path = manifest_path or os.path.join("./results", "manifest.csv")
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{self.name}] manifest -> {manifest_path}")
        return sorted(paths)
 
 
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
 
 
# ==========================================================
# 1. LOCAL — perilaku lama, tetap dipertahankan
# ==========================================================
class LocalSource(BaseSource):
    """Baca PDF dari folder di disk. Ini default, biar nggak ada regresi."""
 
    name = "local"
 
    def __init__(self, pdf_dir: str = "./data/pdf"):
        self.pdf_dir = pdf_dir
 
    def list_documents(self) -> List[RemoteDoc]:
        # Cek folder duluan. Tanpa ini glob() balik [] tanpa suara dan
        # sync() melempar pesan generik "cek nama bucket dan izin akses"
        # yang tidak relevan untuk sumber lokal dan tidak menyebut path
        # mana yang sebenarnya dicari.
        if not os.path.isdir(self.pdf_dir):
            raise RuntimeError(
                f"[{self.name}] folder tidak ditemukan: {os.path.abspath(self.pdf_dir)}\n"
                f"        Siapkan datanya : python scripts/link_pdfs.py\n"
                f'        atau tunjuk lain: python sources.py --pdf-dir "<folder berisi PDF>"'
            )

        out = []
        for p in sorted(glob.glob(os.path.join(self.pdf_dir, "*.pdf"))):
            out.append(
                RemoteDoc(
                    filename=os.path.basename(p),
                    remote_ref=os.path.abspath(p),
                    revision=str(int(os.path.getmtime(p))),
                    size=os.path.getsize(p),
                )
            )
        if not out:
            raise RuntimeError(
                f"[{self.name}] folder ada tapi tidak berisi PDF: "
                f"{os.path.abspath(self.pdf_dir)}\n"
                f"        Siapkan datanya: python scripts/link_pdfs.py"
            )
        return out
 
    def fetch(self, doc: RemoteDoc, cache_dir: str) -> str:
        # File sudah lokal — tidak perlu disalin ke mana-mana.
        return doc.remote_ref
 
 
# ==========================================================
# 2. SUPABASE STORAGE
# ==========================================================
class SupabaseSource(BaseSource):
    """
    Ambil PDF dari Supabase Storage bucket.
 
    pip install supabase
 
    ENV:
      SUPABASE_URL          https://xxxx.supabase.co
      SUPABASE_SERVICE_KEY  service_role key (bucket privat)
                            atau anon key (bucket publik + policy SELECT)
 
    Kalau bucket-nya privat, anon key TIDAK cukup kecuali lo bikin RLS
    policy. Untuk skrip riset yang jalan di laptop sendiri, service_role
    key paling praktis — tapi jangan sekali-kali ditaruh di Streamlit
    yang di-deploy publik.
    """
 
    name = "supabase"
 
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        from supabase import create_client  # import lokal: opsional dependency
 
        url = url or os.getenv("SUPABASE_URL", "")
        key = key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY belum di-set")
 
        self.client = create_client(url, key)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
 
    def list_documents(self) -> List[RemoteDoc]:
        store = self.client.storage.from_(self.bucket)
        docs, offset, page = [], 0, 100
 
        while True:
            batch = store.list(
                path=self.prefix or None,
                options={"limit": page, "offset": offset, "sortBy": {"column": "name",
                                                                     "order": "asc"}},
            )
            if not batch:
                break
            for item in batch:
                nm = item.get("name", "")
                if not nm.lower().endswith(".pdf"):
                    continue  # lewati folder & file non-PDF
                meta = item.get("metadata") or {}
                docs.append(
                    RemoteDoc(
                        filename=nm,
                        remote_ref=f"{self.prefix}/{nm}" if self.prefix else nm,
                        # eTag berubah setiap file di-upload ulang
                        revision=str(meta.get("eTag") or item.get("updated_at") or ""),
                        size=meta.get("size"),
                    )
                )
            if len(batch) < page:
                break
            offset += page
        return docs
 
    def _read_bytes(self, doc: RemoteDoc) -> bytes:
        return self.client.storage.from_(self.bucket).download(doc.remote_ref)
 
 
# ==========================================================
# 3. GOOGLE DRIVE (service account)
# ==========================================================
class GoogleDriveSource(BaseSource):
    """
    Ambil PDF dari satu folder Google Drive.
 
    pip install google-api-python-client google-auth
 
    Setup sekali:
      1. console.cloud.google.com -> buat project -> aktifkan Drive API
      2. Buat Service Account -> unduh JSON key
      3. Buka folder Drive-nya -> Share -> masukkan email service account
         (bentuknya xxx@xxx.iam.gserviceaccount.com) sebagai Viewer
      4. folder_id = potongan URL setelah /folders/
 
    Service account dipilih ketimbang OAuth user karena skrip batch
    nggak boleh butuh klik "Allow" di browser tiap kali jalan.
 
    ENV:
      GDRIVE_SERVICE_ACCOUNT_JSON  path ke file JSON key
    """
 
    name = "gdrive"
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
 
    def __init__(self, folder_id: str, credentials_path: Optional[str] = None):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
 
        cred_path = credentials_path or os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "")
        if not cred_path or not os.path.exists(cred_path):
            raise RuntimeError("GDRIVE_SERVICE_ACCOUNT_JSON tidak ditemukan")
 
        creds = service_account.Credentials.from_service_account_file(
            cred_path, scopes=self.SCOPES
        )
        self.service = build("drive", "v3", credentials=creds)
        self.folder_id = folder_id
 
    def list_documents(self) -> List[RemoteDoc]:
        q = (
            f"'{self.folder_id}' in parents "
            "and mimeType='application/pdf' and trashed=false"
        )
        docs, token = [], None
        while True:
            resp = (
                self.service.files()
                .list(
                    q=q,
                    fields="nextPageToken, files(id,name,md5Checksum,size,modifiedTime)",
                    pageSize=200,
                    pageToken=token,
                    # dua baris ini wajib kalau foldernya ada di Shared Drive
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in resp.get("files", []):
                docs.append(
                    RemoteDoc(
                        filename=f["name"],
                        remote_ref=f["id"],
                        revision=f.get("md5Checksum") or f.get("modifiedTime", ""),
                        size=int(f["size"]) if f.get("size") else None,
                    )
                )
            token = resp.get("nextPageToken")
            if not token:
                break
        return docs
 
    def _read_bytes(self, doc: RemoteDoc) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload
 
        req = self.service.files().get_media(fileId=doc.remote_ref)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()
 
 
# ==========================================================
# 4. SHAREPOINT / ONEDRIVE (Microsoft Graph)
# ==========================================================
class SharePointSource(BaseSource):
    """
    Ambil PDF dari document library SharePoint lewat Microsoft Graph.
 
    pip install msal requests
 
    Ini yang paling ribet setup-nya karena butuh admin tenant:
      1. Azure Portal -> App registrations -> New registration
      2. Certificates & secrets -> New client secret
      3. API permissions -> Microsoft Graph -> Application permissions
         -> Sites.Read.All -> KLIK "Grant admin consent"
      4. site_id: buka
         https://graph.microsoft.com/v1.0/sites/{host}:/sites/{nama-site}
         lewat Graph Explorer, ambil field "id"
 
    JALAN PINTAS: kalau lo nggak punya akses admin tenant (sering terjadi
    di kampus/kantor), sync foldernya pakai aplikasi OneDrive desktop,
    lalu pakai LocalSource ke folder hasil sync. Hasil akhirnya identik
    dan lo hemat dua minggu nunggu approval IT.
 
    ENV:
      MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET
    """
 
    name = "sharepoint"
 
    def __init__(self, site_id: str, folder_path: str = "", drive_id: str = ""):
        import msal
 
        tenant = os.getenv("MS_TENANT_ID", "")
        client_id = os.getenv("MS_CLIENT_ID", "")
        secret = os.getenv("MS_CLIENT_SECRET", "")
        if not all([tenant, client_id, secret]):
            raise RuntimeError("MS_TENANT_ID / MS_CLIENT_ID / MS_CLIENT_SECRET belum di-set")
 
        app = msal.ConfidentialClientApplication(
            client_id,
            authority=f"https://login.microsoftonline.com/{tenant}",
            client_credential=secret,
        )
        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        if "access_token" not in result:
            raise RuntimeError(f"Gagal ambil token: {result.get('error_description')}")
 
        self.token = result["access_token"]
        self.site_id = site_id
        self.drive_id = drive_id
        self.folder_path = folder_path.strip("/")
 
    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}
 
    def _drive_root(self) -> str:
        base = "https://graph.microsoft.com/v1.0"
        if self.drive_id:
            return f"{base}/drives/{self.drive_id}"
        return f"{base}/sites/{self.site_id}/drive"
 
    def list_documents(self) -> List[RemoteDoc]:
        import requests
 
        root = self._drive_root()
        url = (
            f"{root}/root:/{self.folder_path}:/children"
            if self.folder_path
            else f"{root}/root/children"
        )
 
        docs = []
        while url:
            r = requests.get(url, headers=self._headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            for item in data.get("value", []):
                nm = item.get("name", "")
                if not nm.lower().endswith(".pdf"):
                    continue
                docs.append(
                    RemoteDoc(
                        filename=nm,
                        # downloadUrl sudah pre-signed, jadi disimpan langsung.
                        # Catatan: URL ini kedaluwarsa ~1 jam, jadi jangan
                        # list dulu lalu unduh besok.
                        remote_ref=item.get("@microsoft.graph.downloadUrl", ""),
                        revision=item.get("eTag", "") or item.get("lastModifiedDateTime", ""),
                        size=item.get("size"),
                    )
                )
            url = data.get("@odata.nextLink")
        return docs
 
    def _read_bytes(self, doc: RemoteDoc) -> bytes:
        import requests
 
        r = requests.get(doc.remote_ref, timeout=300)
        r.raise_for_status()
        return r.content
 
 
# ==========================================================
# Factory — dipanggil dari runner.py
# ==========================================================
def build_source(args) -> BaseSource:
    """Terjemahkan argumen CLI jadi objek source."""
    s = args.source
    if s == "local":
        return LocalSource(pdf_dir=args.pdf_dir)
    if s == "supabase":
        return SupabaseSource(bucket=args.bucket, prefix=args.prefix)
    if s == "gdrive":
        return GoogleDriveSource(folder_id=args.folder_id)
    if s == "sharepoint":
        return SharePointSource(site_id=args.site_id, folder_path=args.prefix)
    raise ValueError(f"source tidak dikenal: {s}")
 
 
# ==========================================================
# Uji cepat tanpa menyentuh pipeline
# ==========================================================
if __name__ == "__main__":
    import argparse
 
    p = argparse.ArgumentParser(description="Tes koneksi sumber dokumen")
    p.add_argument("--source", default="local",
                   choices=["local", "supabase", "gdrive", "sharepoint"])
    p.add_argument("--pdf-dir", default="./data/pdf")
    p.add_argument("--bucket", default="")
    p.add_argument("--prefix", default="")
    p.add_argument("--folder-id", default="")
    p.add_argument("--site-id", default="")
    p.add_argument("--cache-dir", default="./data/pdf")
    p.add_argument("--list-only", action="store_true",
                   help="cuma tampilkan daftar, jangan unduh")
    args = p.parse_args()
 
    src = build_source(args)
    if args.list_only:
        for d in src.list_documents():
            size = f"{d.size/1e6:.1f} MB" if d.size else "?"
            print(f"  {d.doc_id:<30} {d.filename:<40} {size}")
    else:
        paths = src.sync(cache_dir=args.cache_dir)
        print(f"\n{len(paths)} PDF siap di {args.cache_dir}")
 
