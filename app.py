
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from core.pipeline import run_pipeline, FORMATS
from core.eval_harness import run_evals

st.set_page_config(page_title="Multi-Modal AI Content Platform", page_icon="🎙️", layout="wide")
st.title("🎙️ Multi-Modal AI Content Platform")
st.caption("Image + Voice + Automated content pipeline + Agent tools + Evals")

tab_main, tab_eval = st.tabs(["Create content", "Evaluation harness"])

with st.sidebar:
    fmt = st.selectbox("Output format", list(FORMATS))
    tone = st.selectbox("Tone", ["professional", "casual", "inspiring", "witty"])
    voice_out = st.checkbox("Read result aloud (TTS)", True)

with tab_main:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Image (optional)")
        img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if img:
            st.image(img, use_container_width=True)
        st.subheader("2. Topic")
        topic = st.text_area("Type a topic/instruction (or use voice)", height=80)
    with c2:
        st.subheader("3. Voice (optional)")
        mic = st.audio_input("Record an instruction")
        up = st.file_uploader("...or upload audio", type=["wav", "mp3", "m4a", "ogg"], key="aud")

    if st.button("🚀 Run pipeline", type="primary"):
        audio, mime = None, "audio/wav"
        if mic:
            audio = mic.getvalue()
        elif up:
            audio = up.getvalue()
            mime = up.type or "audio/mpeg"
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                st.session_state.result = run_pipeline(
                    topic, img.getvalue() if img else None, audio, mime,
                    fmt, tone, voice_out)
                status.update(label="Done", state="complete")
            except Exception as e:
                st.session_state.result = {"error": f"Something failed: {e}"}
                status.update(label="Failed", state="error")

    res = st.session_state.get("result")
    if res:
        if "error" in res:
            st.error(res["error"])
        else:
            if "transcript" in res:
                st.info(f"🎤 Heard: {res['transcript']}")
            if "image_analysis" in res:
                ia = res["image_analysis"]
                with st.expander(f"🖼️ Image analysis (confidence: {ia.get('confidence')})"):
                    st.json(ia)
                if ia.get("confidence") == "low" or ia.get("uncertainties"):
                    st.warning("Uncertainties: " + ", ".join(map(str, ia.get("uncertainties", []))))
            if res.get("tool_calls"):
                with st.expander("🛠️ Agent tool calls"):
                    st.json(res["tool_calls"])
            st.subheader("✍️ Final content")
            st.write(res["final"])
            if res.get("audio"):
                st.audio(res["audio"], format="audio/mp3")
            elif res.get("tts_error"):
                st.warning("Voice output unavailable: " + res["tts_error"])
            st.subheader("⏱️ Latency (s)")
            st.table([res["timings"]])

with tab_eval:
    st.write("Runs the agent against a test set and checks tool use, content, and an LLM judge score.")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating..."):
            df = run_evals()
        st.dataframe(df, use_container_width=True)
        st.metric("Avg judge score", round(df["judge_1to5"].mean(), 2))
        st.metric("Tool-use accuracy", f"{df['tool_ok'].mean():.0%}")