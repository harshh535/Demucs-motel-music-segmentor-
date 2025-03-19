import os
import torch
import torchaudio
from pydub import AudioSegment
import streamlit as st
from demucs.pretrained import get_model
from demucs.apply import apply_model
from io import BytesIO
import tempfile
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Ensure ffmpeg is set for Pydub
AudioSegment.converter = "ffmpeg"

# Streamlit cache to load model only once
@st.cache_resource
def load_demucs_model():
    try:
        model_name = 'htdemucs'
        model = get_model(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, device
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")

# Streamlit cache to load torch backend correctly
@st.cache_resource
def setup_torchaudio_backend():
    try:
        torchaudio.set_audio_backend("sox_io")  # Ensures correct backend
    except Exception as e:
        logging.warning(f"Torchaudio backend setup failed: {e}")

setup_torchaudio_backend()

# Convert to WAV using pydub
def convert_to_wav(input_path):
    try:
        if input_path.endswith(".mp3"):
            raise ValueError("Torchaudio does not support MP3 well; using pydub.")

        wav, sr = torchaudio.load(input_path)
        temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        torchaudio.save(temp_wav_path, wav, sr)
        return temp_wav_path
    except Exception as torchaudio_error:
        logging.warning(f"Torchaudio conversion failed: {torchaudio_error}")
        try:
            audio = AudioSegment.from_file(input_path)
            temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio.export(temp_wav_path, format="wav")
            return temp_wav_path
        except Exception as pydub_error:
            raise ValueError(f"Conversion failed: {pydub_error}")

# Load Audio File
def load_audio(file_path, sample_rate=44100):
    try:
        wav, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
        return wav, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")

# Perform Stem Separation
def separate_stems(model, wav, device):
    wav = wav.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        stems = apply_model(model, wav)
    return stems

# Save Stems as Bytes
def save_stems_as_bytes(stems, sample_rate, stem_names):
    stem_files = {}
    for idx, stem in enumerate(stems):
        try:
            if stem.ndimension() == 3:
                for b in range(stem.shape[0]):
                    stem_b = stem[b].squeeze(0)
                    buffer = BytesIO()
                    torchaudio.save(buffer, stem_b.cpu(), sample_rate, format="wav")
                    buffer.seek(0)
                    stem_files[f"{stem_names[idx]}_{b}"] = buffer
            else:
                raise ValueError(f"Unexpected tensor shape: {stem.shape}")
        except Exception as e:
            logging.error(f"Error saving {stem_names[idx]}: {e}")
            continue
    return stem_files

# Streamlit UI
def main():
    st.title("🎵 AI Music Stem Separator")

    uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "flac"], accept_multiple_files=True)

    if uploaded_files:
        model, device = load_demucs_model()

        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_input:
                temp_input.write(uploaded_file.read())
                temp_file_path = temp_input.name

            try:
                wav_file_path = convert_to_wav(temp_file_path)  # Convert to WAV if needed
                wav, sample_rate = load_audio(wav_file_path)

                with st.spinner("🎶 AI is processing your file..."):
                    stems = separate_stems(model, wav, device)

                stem_names = ["drums", "bass", "other", "vocals"]
                stem_files = save_stems_as_bytes(stems, sample_rate, stem_names)

                st.success(f"✅ Stem separation complete for {uploaded_file.name}!")
                for idx, (name, buffer) in enumerate(stem_files.items()):
                    correct_name = stem_names[idx % len(stem_names)]
                    st.audio(buffer, format="audio/wav", start_time=0)
                    st.download_button(f"⬇ Download {correct_name}", buffer, file_name=f"{uploaded_file.name}_{correct_name}.wav")

            except Exception as e:
                st.error(f"❌ Error with {uploaded_file.name}: {e}")
            finally:
                os.unlink(temp_file_path)
                if 'wav_file_path' in locals() and os.path.exists(wav_file_path):
                    os.unlink(wav_file_path)

if __name__ == "__main__":
    main()
