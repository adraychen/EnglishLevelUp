import os
import io
import tempfile
import base64
import streamlit as st
from groq import Groq
from gtts import gTTS
from crewai import Agent, Task, Crew, Process
from audio_recorder_streamlit import audio_recorder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="English Grammar Coach", page_icon="🎓")
st.title("🎓 English Grammar Coach")
st.caption("Answer my question out loud — I'll help you speak more naturally!")

# ── API key ───────────────────────────────────────────────────────────────────
groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("API Key missing! Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key
groq_client = Groq(api_key=groq_key)

# ── Agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return Agent(
        role="English Grammar Coach",
        goal="Help the student speak English more naturally and correctly.",
        backstory="""You are a friendly English teacher.
        When given a sentence, you:
        1. Provide the corrected sentence first, clearly labelled.
        2. Explain ONE key mistake in simple language (1-2 sentences max).
        3. If the sentence is already correct, say 'Great job! That was perfect.'
        Keep your response short and encouraging.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = [
    "Tell me what you did this morning.",
    "Describe your favourite food.",
    "Tell me about your school or workplace.",
    "What do you like to do on weekends?",
    "Describe a place you would like to visit.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=af,
                language="en",
            )
    return result.text.strip()

def correct_grammar(sentence: str) -> str:
    task = Task(
        description=f"Review this sentence for grammar and fluency: \"{sentence}\"",
        expected_output="Corrected sentence + brief explanation of one key mistake (or praise if correct).",
        agent=get_agent(),
    )
    crew = Crew(agents=[get_agent()], tasks=[task], process=Process.sequential)
    return crew.kickoff().raw

def make_audio_b64(text: str) -> str:
    """Generate gTTS audio and return as base64 string."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def autoplay_audio(b64: str):
    """Inject hidden audio tag that autoplays — no visible player."""
    st.markdown(
        f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_index" not in st.session_state:
    st.session_state.prompt_index = 0
if "question_asked" not in st.session_state:
    st.session_state.question_asked = False
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = True

# ── Audio toggle button ───────────────────────────────────────────────────────
col_tog, _ = st.columns([1, 5])
with col_tog:
    label = "🔊 Audio ON" if st.session_state.audio_enabled else "🔇 Audio OFF"
    if st.button(label, use_container_width=True):
        st.session_state.audio_enabled = not st.session_state.audio_enabled
        st.rerun()

st.divider()

# ── Ask the current question ──────────────────────────────────────────────────
current_index = st.session_state.prompt_index
all_done = current_index >= len(PROMPTS)

if not all_done and not st.session_state.question_asked:
    question = PROMPTS[current_index]
    st.session_state.messages.append({
        "role": "assistant",
        "content": question,
        "audio_b64": make_audio_b64(question),
    })
    st.session_state.question_asked = True

# ── Render chat history ───────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Autoplay audio for the latest assistant message only ─────────────────────
if st.session_state.audio_enabled and st.session_state.messages:
    last = st.session_state.messages[-1]
    if last["role"] == "assistant" and "audio_b64" in last:
        autoplay_audio(last["audio_b64"])

# ── All done ──────────────────────────────────────────────────────────────────
if all_done:
    st.success("✅ Great practice! You've completed all 5 questions.")
    if st.button("Start over"):
        st.session_state.messages = []
        st.session_state.prompt_index = 0
        st.session_state.question_asked = False
        st.rerun()
    st.stop()

# ── Progress ──────────────────────────────────────────────────────────────────
st.caption(f"Question {current_index + 1} of {len(PROMPTS)}")
st.progress(current_index / len(PROMPTS))

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("**🎤 Tap to start · tap again to stop**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e85d04",
        neutral_color="#6c757d",
        icon_size="2x",
        pause_threshold=3.0,
        key=f"recorder_{current_index}",
    )
with col2:
    st.markdown("**⌨️ Or type your answer:**")
    text_input = st.chat_input("Type your answer here...")

# ── Process answer ────────────────────────────────────────────────────────────
def handle_answer(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.spinner("Reviewing your sentence..."):
        feedback = correct_grammar(user_text)
    st.session_state.messages.append({
        "role": "assistant",
        "content": feedback,
        "audio_b64": make_audio_b64(feedback),
    })
    st.session_state.prompt_index += 1
    st.session_state.question_asked = False
    st.rerun()

if audio_bytes and len(audio_bytes) > 1000:
    with st.spinner("Transcribing..."):
        try:
            user_text = transcribe(audio_bytes)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            user_text = None
    if user_text:
        handle_answer(user_text)

elif text_input:
    handle_answer(text_input)
