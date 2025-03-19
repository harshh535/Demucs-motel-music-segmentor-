import streamlit as st
import os
import torch
import torchaudio
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.apply import apply_model
from io import BytesIO
import tempfile

# Logo Path
LOGO_PATH = "logo.png"
st.set_page_config(
    page_title="StemFlow", 
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None, 
    layout="wide"
)

# Streamlit UI Styling
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
    """, 
    unsafe_allow_html=True
)

# Homepage UI
def homepage():
    col1, col2 = st.columns([0.7, 0.3])
    with col2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=400)
    
    st.markdown('<h1 class="heading1">StemFlow</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="bodytext">StemFlow uses deep learning to separate music into individual stems.</h2>', unsafe_allow_html=True)

# About Page
def about_stemflow():
    st.markdown('<h1 class="heading1">About StemFlow</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="bodytext">StemFlow is powered by Demucs, a deep learning model developed by Meta.</h2>', unsafe_allow_html=True)
    st.markdown("[Demucs Repository](https://github.com/facebookresearch/demucs)")

# Step 1: Load the Demucs Model
def load_demucs_model():
    try:
        model_name = 'htdemucs'  # Valid model name for 4-stem configuration
        model = get_model(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, device
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")

# Step 2: Convert Audio to WAV if Necessary
def convert_to_wav(input_path):
    try:
        wav, sr = torchaudio.load(input_path)
        temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        torchaudio.save(temp_wav_path, wav, sr)
        return temp_wav_path
    except Exception as torchaudio_error:
        print(f"Torchaudio conversion failed: {torchaudio_error}")
        try:
            audio = AudioSegment.from_file(input_path)
            temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio.export(temp_wav_path, format="wav")
            return temp_wav_path
        except Exception as pydub_error:
            raise ValueError(f"Both torchaudio and pydub failed to convert: {pydub_error}")

# Step 3: Load the Audio File
def load_audio(file_path, sample_rate=44100):
    try:
        wav, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
        return wav, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")

# Step 4: Perform Stem Separation
def separate_stems(model, wav, device):
    wav = wav.unsqueeze(0).to(device)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        stems = apply_model(model, wav)  # Model output
    return stems

# Step 5: Save Stems as Bytes
def save_stems_as_bytes(stems, sample_rate, stem_names):
    stem_files = {}
    for idx, stem in enumerate(stems):
        try:
            if stem.ndimension() == 3:  # Handle the 3D tensor (batch, channels, samples)
                for b in range(stem.shape[0]):
                    stem_b = stem[b].squeeze(0)  # Remove batch dimension
                    buffer = BytesIO()
                    torchaudio.save(buffer, stem_b.cpu(), sample_rate, format="wav")
                    buffer.seek(0)
                    stem_files[f"{stem_names[idx]}_{b}"] = buffer
            else:
                raise ValueError(f"Unexpected tensor shape: {stem.shape}")
        except Exception as e:
            print(f"Error saving {stem_names[idx]}: {e}")
            continue
    return stem_files

# Step 6: Streamlit UI Setup
def main():
    st.title("Effortless Music Demixing Tool")

    # Upload audio files (batch upload)
    uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "flac"], accept_multiple_files=True)

    if uploaded_files:
        # Load Demucs model
        model, device = load_demucs_model()

        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_input:
                temp_input.write(uploaded_file.read())
                temp_file_path = temp_input.name

            try:
                # Convert input file to WAV if necessary
                wav_file_path = convert_to_wav(temp_file_path)

                # Load audio into torch format
                wav, sample_rate = load_audio(wav_file_path)

                # Perform stem separation
                with st.spinner("AI is now processing your file. It may take a while..."):
                    stems = separate_stems(model, wav, device)

                # Stem names for the 4-stem configuration
                stem_names = ["drums", "bass", "other", "vocals"]

                # Save stems and display options
                stem_files = save_stems_as_bytes(stems, sample_rate, stem_names)

                st.success(f"Stem separation complete for {uploaded_file.name}!")
                for idx, (name, buffer) in enumerate(stem_files.items()):
                    correct_name = stem_names[idx % len(stem_names)]
                    st.audio(buffer, format="audio/wav", start_time=0)
                    st.download_button(f"Download {correct_name}", buffer, file_name=f"{uploaded_file.name}_{correct_name}.wav")

            except Exception as e:
                st.error(f"An error occurred with {uploaded_file.name}: {e}")
            finally:
                # Cleanup temporary files
                os.unlink(temp_file_path)
                if 'wav_file_path' in locals() and os.path.exists(wav_file_path):
                    os.unlink(wav_file_path)

# Sidebar Navigation
page = st.sidebar.radio("Navigate:", ("Home", "About StemFlow", "StemFlow Model"))
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH)

if page == "Home":
    homepage()
elif page == "About StemFlow":
    about_stemflow()
elif page == "StemFlow Model":
    main()
