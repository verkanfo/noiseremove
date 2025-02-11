import streamlit as st
import numpy as np
import io
import pydub
from df.enhance import enhance, init_df, save_audio, get_model_basedir
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch

st.title("Noise Remove Online")
st.subheader("Made with ❤ by Verkanfo")

uploaded_file = st.file_uploader("Upload your audio file here")
if uploaded_file is not None:
    audio_array, sample_rate = librosa.load(uploaded_file, sr=None) 
    audio_stream = io.BytesIO(uploaded_file.getvalue())
    audio = pydub.AudioSegment.from_file(audio_stream)
    temp_audio_file = io.BytesIO()
    audio.export(temp_audio_file, format="wav")
    raw_audio = temp_audio_file.getvalue()
    st.audio(raw_audio, format='audio/wav')

    model_option = st.selectbox(
        "Which DeepFilterNet model you want to use?",
        ("DeepFilterNet","DeepFilterNet2","DeepFilterNet3"),
    )

    if model_option:
        if model_option != "DeepFilterNet3":
            model_dir = get_model_basedir(model_option)
            model, df_state, _ = init_df(model_base_dir=model_dir)
        else:
            model, df_state, _ = init_df()

    # print(waveform)
    if st.button('Clean Audio'):
        audio_stream = io.BytesIO(raw_audio) 
        
        waveform, sample_rate = torchaudio.load(audio_stream)
        enhanced = enhance(model, df_state, waveform)
        enhanced_numpy = enhanced.cpu().numpy()
        st.write('Cleaned audio')
        st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)