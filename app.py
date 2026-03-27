import io
import time

import librosa
import matplotlib
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import streamlit as st
import tensorflow as tf

matplotlib.use("Agg")

# Page config
st.set_page_config(
    page_title="Tiny Cat Meow Translator",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Constants — must exactly match training config
DURATION    = 5
SAMPLE_RATE = 16000
TARGET_LEN  = SAMPLE_RATE * DURATION
N_FFT       = 1024
HOP_LENGTH  = 512
N_MELS      = 64
FMIN        = 20
FMAX        = 8_000
CHANNELS    = 3
MODEL_PATH  = "efficientnetb0.keras"
CLASSES     = ["Angry", "Happy", "Paining", "Resting", "Warning"]
MAX_FILE_MB = 10

CLASS_META = {
    "Angry":   {"emoji": "😾", "color": "#E74C3C", "desc": "Your cat is annoyed or threatened."},
    "Happy":   {"emoji": "😸", "color": "#2ECC71", "desc": "Your cat is content and relaxed."},
    "Paining": {"emoji": "🙀", "color": "#E67E22", "desc": "Your cat may be in discomfort."},
    "Resting": {"emoji": "😴", "color": "#3498DB", "desc": "Your cat is calm and at rest."},
    "Warning": {"emoji": "⚠️",  "color": "#9B59B6", "desc": "Your cat is alerting you to something."},
}

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.block-container { padding-top: 1rem !important; }

.hero { text-align: center; padding: 0.5rem 0 1rem; }
.hero h1 { font-size: 2.6rem; line-height: 1.15; color: #1a1a2e; margin-bottom: 0.3rem; }
.hero p  { font-size: 1.05rem; color: #666; font-weight: 300; margin: 0.6rem; }

.result-card { border-radius: 16px; padding: 1.2rem 1.5rem; margin: 1.5rem 0; border-left: 6px solid; background: #fafafa; }
.result-label { font-size: 1.5rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: #999; margin-bottom: 0.3rem; }
.result-emotion { font-family: 'DM Serif Display', serif; font-size: 1rem; margin: 0 0 0.6rem 0; line-height: 1.2; }
.result-desc { font-size: 0.95rem; color: #555; margin-top: 0.5rem; }

.prob-row   { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 0.88rem; }
.prob-label { width: 72px; text-align: right; color: #444; font-weight: 500; flex-shrink: 0; }
.prob-bar-bg   { flex: 1; background: #efefef; border-radius: 999px; height: 10px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }
.prob-pct   { width: 42px; text-align: right; color: #666; font-size: 0.82rem; flex-shrink: 0; }

.upload-hint { text-align: center; font-size: 0.85rem; color: #aaa; margin-top: -0.5rem; margin-bottom: 1rem; }
.spec-caption { text-align: center; font-size: 0.78rem; color: #aaa; margin-top: 0.3rem; }
.conf-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 1rem; font-weight: 500; margin-left: 0.5rem; vertical-align: middle; }

.footer { text-align: center; font-size: 0.9rem; color: #bbb; margin-top: 3rem; padding-bottom: 1rem; }
.footer a { color: #bbb; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# Model
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Audio helpers
def load_and_fix_audio(audio_bytes):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    if len(y) > TARGET_LEN:
        start = (len(y) - TARGET_LEN) // 2
        y = y[start : start + TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    return y


def to_mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)
    mel = mel * 255.0
    return np.repeat(mel[..., np.newaxis], CHANNELS, axis=-1)  # (64, 157, 3)


def run_predict(model, y):
    X = to_mel_spectrogram(y)[np.newaxis]   # (1, 64, 157, 3)
    t0 = time.perf_counter()
    raw = model.predict(X, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = {cls: float(p) for cls, p in zip(CLASSES, raw)}
    return probs, latency_ms


def validate_audio(file):
    if len(file.getvalue()) / 1e6 > MAX_FILE_MB:
        return f"File exceeds {MAX_FILE_MB} MB limit."
    if not file.name.lower().endswith((".wav", ".mp3")):
        return "Unsupported format. Please upload WAV, MP3."
    return None


# UI helpers
def render_probability_bars(probs, top_class):
    bars_html = ""
    for cls in sorted(probs, key=probs.get, reverse=True):
        p     = probs[cls]
        color = CLASS_META[cls]["color"]
        bold  = "font-weight:600;" if cls == top_class else ""
        bars_html += f"""
        <div class="prob-row">
            <div class="prob-label" style="{bold}">{cls}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{color};"></div>
            </div>
            <div class="prob-pct">{p*100:.1f}%</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)


def render_result(probs, latency_ms):
    top_class = max(probs, key=probs.get)
    top_prob  = probs[top_class]
    meta      = CLASS_META[top_class]

    if top_prob >= 0.75:
        conf_label, conf_bg = "High confidence >= 75%", "#d4edda"
    elif top_prob >= 0.50:
        conf_label, conf_bg = "Moderate confidence >= 50%", "#fff3cd"
    else:
        conf_label, conf_bg = "Low confidence < 50%", "#f8d7da"
    
    st.markdown(f"""
    <div class="result-card" style="border-color:{meta['color']};">
        <p style="font-weight:600;margin:0 0 0.6rem 0;">Prediction result</p>
        <div class="result-label">Your cat sounds…</div>
        <p class="result-emotion" style="color:{meta['color']};">
            {meta['emoji']} {top_class}
            <span class="conf-badge" style="background:{conf_bg};color:#333;">{conf_label}</span>
        </p>
        <p class="result-desc">{meta['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Probability Breakdown**")
    render_probability_bars(probs, top_class)
    st.caption(f"Inference time: {latency_ms:.0f} ms · Model: EfficientNet-B0 fine-tuned on mel spectrograms")


def render_visualizations(y):
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("**Audio Visualizations**")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

    # Waveform
    times = np.linspace(0, DURATION, num=len(y))
    ax1.plot(times, y, color="#3498DB", linewidth=0.8)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_title("Audio Waveform", fontsize=10, fontweight="bold")
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.2)

    # Mel-spectrogram
    mel_raw = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    librosa.display.specshow(
        librosa.power_to_db(mel_raw, ref=np.max),
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
        x_axis="time", y_axis="mel",
        ax=ax2, cmap="magma",
    )
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    ax2.set_title("Audio Mel-spectrogram", fontsize=10, fontweight="bold")
    ax2.tick_params(labelsize=8)

    fig.tight_layout(pad=2.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown(
        '<p class="spec-caption">Audio Waveform (amplitude over time) · Audio Mel Spectrogram (log frequency · dB scale)</p>',
        unsafe_allow_html=True,
    )


# Main
def main():
    st.markdown("""
    <div class="hero">
        <h1>🐱 Tiny Cat Meow Translator</h1>
        <p>Upload a cat sound clip and discover what your cat is feeling.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

    uploaded = st.file_uploader(
        "Upload a cat audio clip",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-hint">WAV · MP3 &nbsp;|&nbsp; max 10 MB</p>',
        unsafe_allow_html=True,
    )

    if uploaded is None:
        st.markdown("---")
        st.markdown("#### How it works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Upload**\nDrop any short cat audio clip (WAV or MP3 recommended).")
        with col2:
            st.markdown("**2. Process**\nThe audio is converted to a mel-spectrogram — a visual fingerprint of sound.")
        with col3:
            st.markdown("**3. Classify**\nEfficientNet-B0, fine-tuned on 5 cat emotion d classes, predicts the emotion.")

        st.markdown("---")
        st.markdown("#### Emotion classes")
        cols = st.columns(len(CLASSES))
        for col, cls in zip(cols, CLASSES):
            m = CLASS_META[cls]
            col.markdown(f"**{m['emoji']} {cls}**  \n{m['desc']}")
        return

    error = validate_audio(uploaded)
    if error:
        st.error(error)
        return

    st.audio(uploaded.read(), format=uploaded.type)
    uploaded.seek(0)

    with st.spinner("Analysing…"):
        try:
            y = load_and_fix_audio(uploaded.read())
            probs, latency_ms = run_predict(model, y)
        except Exception as e:
            st.error(f"Something went wrong during analysis: {e}")
            return

    render_result(probs, latency_ms)

    render_visualizations(y)

    st.markdown("""
    <div class="footer">
    <p>Audio classifier of cat meows (Happy, Paining, Angry, Warning, Resting) · Model: EfficientNet-B0 on mel spectrograms</p>
    <p>
        <a href="https://github.com/TeeNguyenDA/Tiny-Cat-Meow-Translator" target="_blank">GitHub</a>
        ·
        <a href="https://www.kaggle.com/datasets/yagtapandeya/cat-sound-classification-dataset/data?select=CAT_DB" target="_blank">CAT_DB dataset</a>
    </p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
