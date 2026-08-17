"""
Interface VERIFIED via Streamlit.
Unique / Novelty yang membuat VERIFIED berbeda dari chatbot PDF:
  1. Confidence judgement, Kalau sistem ragu, dia BILANG ragu, dan
     menyebutkan persis apa yang perlu dicek manusia.
  2. Setiap sitasi bisa dibuka -> teks sumbernya muncul apa adanya.
     Itu traceability yang bisa disentuh, bukan klaim.
  3. Biaya token tampil real-time, jadi pengguna sadar konsekuensi
     sesi panjang.

cmd -> streamlit run app.py
"""

from __future__ import annotations
from config import (
    CONTEXT, EVAL_QUESTIONS, LOG_FILE, MODEL, RETRIEVAL, SOCIOTECHNICAL, VERIFICATION,
)
import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv(override=True)   # .env selalu menang → ganti key cukup restart, tak ada key basi
HOST_ANTHROPIC = os.getenv("ANTHROPIC_API_KEY", "")
HOST_VOYAGE = os.getenv("VOYAGE_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

from typing import Dict, List
st.set_page_config(page_title="VERIFIED Framework", page_icon="🔍", layout="wide")


# ==========================================================
# State
# If Run pertama, maka pakai init_state, nextnya udah disimpen sesison_State dan bisa dipakai session nya
def init_state() -> None:
    defaults = {
        "retriever": None, 
        "pipeline": None, 
        "kg": None,
        "history": [], 
        "doc_id": None, 
        "session_tokens": 0,
        "session_cost": 0.0, 
        "chunk_lookup": {},
    }
    for k, v in defaults.items(): #key, value
        st.session_state.setdefault(k, v)
init_state()


# Pulihkan dokumen aktif saat refresh: session_state hilang saat refresh penuh,
# tapi st.query_params ikut di URL → dipakai reload index dari disk (tanpa embed
# ulang). Kartu file_uploader tak bisa diisi ulang programatik, jadi kita pulihkan
# KONTEKS dokumen + tampilkan indikator.
def restore_doc_from_cache() -> None:
    if st.session_state.retriever is not None:
        return
    doc = st.query_params.get("doc")
    if not doc:
        return
    meta = os.path.join(RETRIEVAL.index_dir, doc + "_chunks.pkl")
    if not os.path.exists(meta):
        return
    try:
        from layer1_retrieval import RetrievalLayer
        r = RetrievalLayer(os.path.join(RETRIEVAL.pdf_dir, doc + ".pdf")).build()
        st.session_state.retriever = r
        st.session_state.doc_id = doc
        st.session_state.chunk_lookup = {c.chunk_id: c for c in r.chunks}
    except Exception:  # noqa: BLE001
        pass
restore_doc_from_cache()


# ==========================================================
# Sidebar
with st.sidebar:
    st.title("🔍 VERIFIED")
    st.caption("Sociotechnical context engineering untuk analisis laporan keuangan IDX30")

    st.subheader("Credentials")
    _env_ok = bool(os.getenv("ANTHROPIC_API_KEY")) and bool(os.getenv("VOYAGE_API_KEY"))
    st.caption("🔑 API key: default menggunakan key evaluator dari .env" if _env_ok
               else "⚠️ API key belum lengkap di .env — isi manual di bawah")
    anthropic_key = voyage_key = ""
    with st.expander("Pakai API key sendiri"):
        anthropic_key = st.text_input(
            "Anthropic API Key", type="password", value="",
            placeholder="⚠️Default akan menggunakan .env⚠️, silahkan isi key mu",
        )
        voyage_key = st.text_input(
            "Voyage API Key", type="password", value="",
            placeholder="⚠️Default akan menggunakan .env⚠️, silahkan isi key mu",
        )
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if voyage_key:
        os.environ["VOYAGE_API_KEY"] = voyage_key

    st.subheader("Scenarios running")
    scenario = st.radio(
        "Mode", ["verified", "standard_rag", "vanilla"],
        format_func=lambda s: {
            "verified": "VERIFIED Framework",
            "standard_rag": "Standard RAG",
            "vanilla": "Vanilla LLM",
        }[s],
    )

    st.subheader("Parameters (If Needed)")
    MODEL.generator_model = st.selectbox(
        "Model", ["claude-sonnet-4-6", "claude-opus-4-8"],
    )
    MODEL.critic_model = MODEL.generator_model
    # RETRIEVAL.top_k_prefetch = st.slider("Prefetch chunk", 1, 10, RETRIEVAL.top_k_prefetch)
    # CONTEXT.max_chunks = st.slider("Chunk ke context", 2, 15, CONTEXT.max_chunks)
    CONTEXT.max_context_tokens = st.slider(
        "Budget token context", 1000, 10000, CONTEXT.max_context_tokens, step = 250
    )

    with st.expander("⚙️ Advanced Settings", expanded=False):
        # VERIFICATION.enable_numerical_gate = st.checkbox(
        #     "Numerical Verification", VERIFICATION.enable_numerical_gate)
        # VERIFICATION.enable_adversarial_critique = st.checkbox(
        #     "Adversasrial Critique", VERIFICATION.enable_adversarial_critique)
        # SOCIOTECHNICAL.enable_human_trigger = st.checkbox(
        #     "Sociotechnical: human judgement", SOCIOTECHNICAL.enable_human_trigger)
        # CONTEXT.reorder_lost_in_the_middle = st.checkbox(
        #     "Context Engineering", CONTEXT.reorder_lost_in_the_middle)
        alpha = st.slider(
            "Similarity vs Tag Match",
            0.0, 1.0,
            value=CONTEXT.weight_similarity,
            step=0.05,
            help="0 = full tag match, 1 = full similarity semantik",
        )
        CONTEXT.weight_similarity = alpha
        CONTEXT.weight_tag_match = round(1 - alpha, 2)
        st.caption(f"weight_similarity = {CONTEXT.weight_similarity} · weight_tag_match = {CONTEXT.weight_tag_match}")

    if st.session_state.session_tokens:
        st.divider()
        st.metric("Token sesi ini", f"{st.session_state.session_tokens:,}")
        st.metric("Biaya sesi ini", f"${st.session_state.session_cost:.4f}")
    # if st.button("Reset sesi"):
    #     st.session_state.history = []
    #     st.session_state.session_tokens = 0
    #     st.session_state.session_cost = 0.0   # WAJIB float, bukan None (dipakai += cost_usd)
    #     st.session_state.pipeline = None
    #     st.rerun()

# ==========================================================
# Upload & index
st.title("Analisis Laporan Keuangan Tahunan")

if scenario == "vanilla":
    st.info("Scenario tidak memerlukan dokumen")
    uploaded, build_kg = None, False
else:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader("Unggah laporan keuangan tahunan (PDF)", type="pdf")
    with col_b:
        build_kg = st.checkbox(
            "Bangun knowledge graph", value=False,
            help="Ekstrak entitas & relasi antar-angka di laporan supaya sistem "
                 "bisa menelusuri hubungan antar-pos. Menambah waktu & biaya pemrosesan.",
        )
    if uploaded is None and st.session_state.retriever is not None:
        st.caption(f"📎 Dokumen aktif dari sesi sebelumnya: **{st.session_state.doc_id}** "
                   "— unggah PDF lain untuk mengganti.")

# ==========================================================
#Case Vanilla
if uploaded and scenario != "vanilla":
    doc_id = os.path.splitext(uploaded.name)[0]
    if st.session_state.doc_id != doc_id:
        os.makedirs(RETRIEVAL.pdf_dir, exist_ok=True)
        path = os.path.join(RETRIEVAL.pdf_dir, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Layer 1: memecah dokumen dan membangun index..."):
            from layer1_retrieval import RetrievalLayer

            retriever = RetrievalLayer(path, embedding_provider="voyage").build()
            st.session_state.retriever = retriever
            st.session_state.doc_id = doc_id
            st.session_state.chunk_lookup = {c.chunk_id: c for c in retriever.chunks}
            st.session_state.pipeline = None
            st.query_params["doc"] = doc_id   # ingat dokumen aktif utk pulih saat refresh

        if build_kg:
            with st.spinner("Layer 1: mengekstraksi knowledge graph..."):
                from layer1_retrieval import KnowledgeGraph
                from llm import LLMClient

                kg = KnowledgeGraph()
                kg.extract_from_chunks(retriever.chunks, LLMClient())
                st.session_state.kg = kg

        st.success(f"{doc_id}: {len(retriever.chunks)} chunk siap dianalisis.")

if st.session_state.retriever and scenario != "vanilla":
    r = st.session_state.retriever
    c1, c2, c3 = st.columns(3)

    def _mini_metric(col, label, value):
        # metrik header dgn value font lebih kecil — cegah doc_id panjang overflow
        col.markdown(
            f"<div style='line-height:1.15'>"
            f"<div style='font-size:0.80rem;color:#9aa0a6'>{label}</div>"
            f"<div style='font-size:1.05rem;font-weight:600'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _mini_metric(c1, "Dokumen", st.session_state.doc_id)
    _mini_metric(c2, "Chunk", len(r.chunks))
    _mini_metric(c3, "Chunk size", RETRIEVAL.chunk_size)


# ==========================================================
# Pertanyaan
st.divider() 
st.subheader("Ajukan pertanyaan")

preset = st.selectbox(
    "Pertanyaan yang digunakan dalam pengujian (atau tulis sendiri apabila ingin mencoba)",
    ["— tulis sendiri —"] + [f"{q['id']}: {q['text']}" for q in EVAL_QUESTIONS],
)
if preset.startswith("—"):
    # tulis sendiri: textarea aktif & kosong
    question = st.text_area("Pertanyaan", value="", height=80)
else:
    # preset dipilih: textarea read-only (tetap terbaca), value dikunci ke teks preset
    preset_text = preset.split(": ", 1)[1]
    st.text_area("Pertanyaan", value=preset_text, height=80, disabled=True)
    question = preset_text

# --- Level risiko (input manual) — dinonaktifkan: konsep internal, membingungkan
#     user awam. Risiko per-pertanyaan tetap jalan via metadata EVAL_QUESTIONS.
#     Aktifkan lagi blok ini + baris risk_override di pipeline.run() bila diperlukan.
# qcol1, qcol2 = st.columns([1, 3])
# with qcol1:
#     risk_override = st.selectbox("Level risiko", ["auto", "low", "medium", "high"])
# with qcol2:
#     st.caption(
#         "Level risiko menentukan seberapa tinggi confidence yang dibutuhkan "
#         "sistem sebelum boleh menjawab tanpa verifikasi manusia."
#     )

if st.button("Analisis", type="primary", disabled=not question.strip()):
    if scenario != "vanilla" and not st.session_state.retriever:
        st.error("Unggah PDF dulu untuk mode ini.")
    else:
        try:
            from pipeline import build_pipeline

            if st.session_state.pipeline is None:
                kwargs = {"kg": st.session_state.kg} if (
                    scenario == "verified" and st.session_state.kg
                ) else {}
                st.session_state.pipeline = build_pipeline(
                    scenario, retriever=st.session_state.retriever, **kwargs
                )

            qmeta = next(
                (q for q in EVAL_QUESTIONS if q["text"] == question),
                {"id": "custom", "risk": "medium"},
            )

            # --- Debate transcript LIVE (sementara) ---
            # Ditulis bertahap tiap agen selesai (Scorer→Critic→Commander) ke satu
            # placeholder, lalu dibersihkan setelah hasil final ada. Datanya hanya
            # lewat callback — tidak disimpan ke PipelineResult.
            debate_ph = st.empty()
            _debate_steps: Dict[str, object] = {}
            _AGENTS = {
                "scorer": ("🧮", "Scorer"),
                "critic": ("😈", "Critic"),
                "commander": ("🫡", "Commander"),
            }

            def on_debate_step(agent, data):
                _debate_steps[agent] = data
                with debate_ph.container():
                    st.caption("🔍 Multiagent debate berlangsung…")
                    for a, (avatar, name) in _AGENTS.items():
                        if a not in _debate_steps:
                            continue
                        with st.chat_message(name, avatar=avatar):
                            d = _debate_steps[a]
                            if d is None:
                                st.markdown(f"**{name}** sedang berpikir… ⏳")
                            elif a == "scorer":
                                st.markdown(f"**Scorer** · confidence `{d['confidence']:.2f}`")
                                if d.get("reasoning"):
                                    st.caption(d["reasoning"])
                            elif a == "critic":
                                st.markdown(f"**Critic** · severity `{str(d.get('severity','none')).upper()}`")
                                for i in d.get("issues", []):
                                    st.markdown(f"- {i}")
                                if not d.get("issues"):
                                    st.caption("Tidak menemukan cacat berarti.")
                            elif a == "commander":
                                st.markdown(
                                    f"**Commander** · verdict `{d.get('verdict','kept')}`"
                                    + (" (jawaban direvisi)" if d.get("revised") else "")
                                )
                                if d.get("suggested_fix"):
                                    st.caption(f"Saran: {d['suggested_fix']}")

            with st.spinner("Menjalankan pipeline..."):
                result = st.session_state.pipeline.run(
                    question=question,
                    question_id=qmeta["id"],
                    question_risk=qmeta.get("risk", "medium"),
                    risk_override=None,  # UI "Level risiko" dinonaktifkan; ganti ke `None if risk_override == "auto" else risk_override` bila diaktifkan lagi
                    doc_id=st.session_state.doc_id or "",
                    company_hint=st.session_state.doc_id or "",
                    on_step=on_debate_step,
                )
            debate_ph.empty()  # bersihkan transkrip sementara; hasil final tampil di bawah
            st.session_state.history.append(result)
            st.session_state.session_tokens += result.total_tokens
            st.session_state.session_cost += result.cost_usd

            # Persist SETIAP run ke results/run_log.csv — skema & kolom anotasi
            # sama persis dengan runner.py (ResultLogger), jadi evaluation.py
            # bisa langsung membacanya bareng hasil batch.
            try:
                from runner import ResultLogger
                if st.session_state.get("logger") is None:
                    st.session_state.logger = ResultLogger()
                st.session_state.logger.write(result)
            except Exception as log_err:  # noqa: BLE001
                st.warning(f"Run berhasil tapi gagal menulis ke {LOG_FILE}: {log_err}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Gagal: {e}")
            st.exception(e)


# ==========================================================
# Tampilan hasil
# ==========================================================
def render_result(res, chunk_lookup: Dict) -> None:
    # 1) JAWABAN DULU
    st.markdown("### Jawaban")
    st.markdown(res.answer)

    # 2) Metrik + badge status ringkas di dekat Confidence
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{res.confidence:.2f}")
    m2.metric("Angka terverifikasi", f"{res.numbers_verified}/{res.numbers_total}")
    m3.metric("Token", f"{res.total_tokens:,}")
    m4.metric("Latensi", f"{res.latency_s}s")
    if res.needs_human:
        icon = "🚨" if res.human_action == "escalate" else "⚠️"
        thr = SOCIOTECHNICAL.confidence_threshold_by_risk.get(res.risk_level, 0.75)
        m1.caption(f"{icon} Human Judgement Required · risk `{res.risk_level}` · ambang {thr:.2f}")
    elif res.scenario == "verified":
        m1.caption("✅ Lolos verifikasi otomatis")

    # 3) PROSES (self-critique & debate) → expander TERTUTUP.
    #    Toggling expander tidak memicu analisis ulang: hasil ada di session_state.
    with st.expander("🔍 Process Self Critqiue  DEBATE", expanded=False):
        if res.human_prompts:
            st.markdown("**Yang perlu dicek manusia:**")
            for p in res.human_prompts:
                st.markdown(f"- {p}")
        st.write(f"**Gate numerik:** {'lolos' if res.gate_passed else 'GAGAL'}")
        st.write(f"**Angka tak terdukung:** {res.numbers_unsupported}")
        st.write(f"**Putusan Commander:** `{res.verdict}`"
                 + (" (jawaban direvisi)" if res.revised else ""))
        if res.critic_issues:
            st.markdown("**Temuan Critic:**")
            for i in res.critic_issues:
                st.markdown(f"- {i}")
        if res.trigger_reasons:
            st.markdown("**Alasan Layer 4:**")
            for r_ in res.trigger_reasons:
                st.markdown(f"- {r_}")

    # 4) Tab referensi (tanpa 'Verifikasi' — sudah dipindah ke expander)
    tabs = st.tabs(["Sumber", "Konteks", "Biaya"])

    with tabs[0]:
        if not res.citations:
            st.info("Jawaban ini tidak memuat sitasi.")
        for cid in res.citations:
            chunk = chunk_lookup.get(cid)
            with st.expander(f"{cid}" + (f" — hal. {chunk.page}" if chunk else "")):
                st.text(chunk.text if chunk else "Teks sumber tidak tersedia.")
                if chunk and chunk.tags:
                    st.caption("Tag: " + ", ".join(chunk.tags))

    with tabs[1]:
        st.write(f"**Tag terdeteksi pada query:** {', '.join(res.query_tags) or '—'}")
        st.write(f"**Chunk dipakai:** {len(res.retrieved_chunks)}")
        st.code("\n".join(res.retrieved_chunks) or "—")

    with tabs[2]:
        st.write(f"Input: {res.input_tokens:,} token")
        st.write(f"Output: {res.output_tokens:,} token")
        st.write(f"Panggilan LLM: {res.llm_calls}")
        st.write(f"Biaya: ${res.cost_usd:.5f}")
        st.caption(
            "VERIFIED memanggil model beberapa kali per pertanyaan "
            "(generate → scorer → critic → commander). Biaya per pertanyaan "
            "memang lebih tinggi; yang lebih landai adalah pertumbuhannya "
            "sepanjang sesi."
        )

if st.session_state.history:
    st.divider()
    st.subheader("Hasil")
    for res in reversed(st.session_state.history):
        with st.container(border=True):
            st.caption(f"`{res.scenario}` · {res.question_id} · {res.question[:90]}")
            render_result(res, st.session_state.chunk_lookup)

    import pandas as pd

    df = pd.DataFrame([r.to_row() for r in st.session_state.history])
    st.download_button(
        "Unduh log sesi (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="verified_session_log.csv",
        mime="text/csv",
    )
    st.caption(f"📝 Setiap analisis juga tersimpan permanen ke `{LOG_FILE}` "
               "(skema sama dengan runner batch → siap dibaca evaluation.py).")
