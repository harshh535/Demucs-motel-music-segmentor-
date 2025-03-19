import streamlit as st
import os
import torch
import torchaudio
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.apply import apply_model
from io import BytesIO
import tempfile
LOGO_PATH = "logo.png"

st.set_page_config(page_title="StemFlow", page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None, layout="wide")

st.markdown(
    """
    <style>
    h1.heading1 {
        color: rgba(187, 47, 133, 1);
        text-align: center;
        font-size: 100px;
        font-family: 'Varela Round', Rubik, Quicksand, sans-serif;
        text-shadow: 6px 6px 10px rgba(37, 0, 100, 0.8);
        margin-top: 20px;
        padding: 10px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def homepage():
    col1, col2 = st.columns([0.7, 0.3])
    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=400)
    st.markdown('<h1 class="heading1">StemFlow</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="bodytext">StemFlow uses deep learning to separate music into individual stems.</h2>', unsafe_allow_html=True)

def about_stemflow():
    st.markdown('<h1 class="heading1">About StemFlow</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="bodytext">StemFlow is powered by Demucs, a deep learning model developed by Meta.</h2>', unsafe_allow_html=True)
    st.markdown("[Demucs Repository](https://github.com/facebookresearch/demucs)")

def load_demucs_model():
    try:
        model = get_model('htdemucs')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def convert_to_wav(input_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError("Input file not found.")
        audio = AudioSegment.from_file(input_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return None

def load_audio(file_path, sample_rate=44100):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("Audio file not found.")
        wav, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
        return wav, sample_rate
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

def separate_stems(model, wav):
    try:
        wav = wav.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            stems = apply_model(model, wav)
        return stems
    except Exception as e:
        st.error(f"Error during separation: {e}")
        return None

def save_stems_as_bytes(stems, sample_rate, stem_names):
    stem_files = {}
    for idx, stem in enumerate(stems):
        try:
            buffer = BytesIO()
            torchaudio.save(buffer, stem.cpu(), sample_rate, format="wav")
            buffer.seek(0)
            stem_files[stem_names[idx]] = buffer
        except Exception as e:
            st.error(f"Error saving {stem_names[idx]}: {e}")
    return stem_files

def model_page():
    st.title("Effortless Music Demixing Tool")
    uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "flac"], accept_multiple_files=True)
    
    if uploaded_files:
        model = load_demucs_model()
        if model is None:
            return

        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            try:
                wav_file_path = convert_to_wav(temp_file_path)
                if not wav_file_path:
                    continue

                wav, sample_rate = load_audio(wav_file_path)
                if wav is None:
                    continue

                with st.spinner("AI is processing your file..."):
                    stems = separate_stems(model, wav)
                    if stems is None:
                        continue

                stem_names = ["drums", "bass", "other", "vocals"]
                stem_files = save_stems_as_bytes(stems, sample_rate, stem_names)

                st.success(f"Stem separation complete for {uploaded_file.name}!")
                for idx, (name, buffer) in enumerate(stem_files.items()):
                    correct_name = stem_names[idx % len(stem_names)]
                    st.audio(buffer, format="audio/wav", start_time=0)
                    st.download_button(f"Download {correct_name}", buffer, file_name=f"{uploaded_file.name}_{correct_name}.wav")
            
            except Exception as e:
                st.error(f"An error occurred with {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if 'wav_file_path' in locals() and wav_file_path and os.path.exists(wav_file_path):
                    os.unlink(wav_file_path)

page = st.sidebar.radio("Navigate:", ("Home", "About StemFlow", "StemFlow Model"))
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH)

if page == "Home":
    homepage()
elif page == "About StemFlow":
    about_stemflow()
elif page == "StemFlow Model":
    model_page()
