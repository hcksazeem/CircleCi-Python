import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.io import wavfile
import io

# --- 1. THE SIGNAL & ATTENTION ENGINE (Wireless OFDM Lab) ---
class OFDMTransformerEngine:
    def __init__(self, sc=96, ts=14):
        self.sc = sc
        self.ts = ts
        self.reset()

    def reset(self):
        self.loss_history = []

    def get_ofdm_frame(self):
        real = np.random.choice([-1.0, 1.0], size=(self.sc, self.ts))
        imag = np.random.choice([-1.0, 1.0], size=(self.sc, self.ts))
        return np.stack([real, imag], axis=-1)

    def apply_idft(self, frame):
        complex_frame = frame[..., 0] + 1j * frame[..., 1]
        time_data = np.fft.ifft(complex_frame, axis=0)
        return np.stack([time_data.real, time_data.imag], axis=-1)

    def inject_noise(self, frame, noise_level):
        noise = np.random.normal(0, noise_level, frame.shape)
        return frame + noise

    def apply_dft(self, noisy_time_frame):
        complex_time = noisy_time_frame[..., 0] + 1j * noisy_time_frame[..., 1]
        symbol_data = np.fft.fft(complex_time, axis=0)
        return np.stack([symbol_data.real, symbol_data.imag], axis=-1)

    def transformer_denoise(self, noisy_frame, learning_progress=0.4):
        ideal_points = np.sign(noisy_frame)
        recovered = (1 - learning_progress) * noisy_frame + learning_progress * ideal_points
        return recovered

    def calculate_metrics(self, original, processed):
        orig_bits = (original > 0).astype(int)
        proc_bits = (processed > 0).astype(int)
        ber = (np.sum(orig_bits != proc_bits) / original.size) * 100

        mse = np.mean((original - processed) ** 2)
        signal_power = np.mean(original ** 2)

        if signal_power == 0:
            return ber, 0.0

        nmse_linear = mse / signal_power
        nmse_db = 10 * np.log10(nmse_linear) if nmse_linear > 0 else -100

        return ber, nmse_db


# --- 2. VOICE-OVER-OFDM DEEP-LEARNING CHANNEL ESTIMATOR ---
class VoiceOFDMChannelEstimator:
    """
    Simulates sending an audio signal through a multipath wireless channel using
    an OFDM-style block transform (frame -> FFT -> subcarriers), then recovers
    the original audio with pilot-aided channel estimation refined by an
    attention (transformer-style) interpolation step across the time/frame axis.

    Pipeline:
      clean audio -> framed -> FFT (subcarriers)
                  -> multiplied by true channel response H_true + AWGN  (TX/channel)
      received subcarriers -> LS estimate of H at known pilot frames
                            -> attention-weighted interpolation across all frames
                            -> zero-forcing equalization
                            -> IFFT -> recovered audio
    """

    def __init__(self, frame_len=128, pilot_spacing=8):
        self.frame_len = frame_len
        self.pilot_spacing = pilot_spacing

    # ---- framing / transform ----
    def frame_signal(self, audio):
        n = len(audio)
        pad = (-n) % self.frame_len
        padded = np.pad(audio, (0, pad))
        return padded.reshape(-1, self.frame_len)

    def to_freq(self, frames):
        return np.fft.fft(frames, axis=1)

    def to_time(self, freq_frames, original_len):
        time_frames = np.fft.ifft(freq_frames, axis=1).real
        flat = time_frames.flatten()
        return flat[:original_len]

    # ---- channel simulation ----
    def random_channel_taps(self, n_taps=5):
        decay = np.exp(-np.arange(n_taps) / 1.5)
        taps = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * decay
        taps[0] = np.abs(taps[0]) + 0.5
        taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
        return taps

    def channel_freq_response(self, taps):
        return np.fft.fft(taps, n=self.frame_len)

    def apply_channel(self, freq_frames, H_true, noise_level):
        distorted = freq_frames * H_true[np.newaxis, :]
        noise = (np.random.normal(0, noise_level, distorted.shape) +
                 1j * np.random.normal(0, noise_level, distorted.shape))
        return distorted + noise

    # ---- pilot-aided estimation ----
    def pilot_indices(self, n_frames):
        idx = np.arange(0, n_frames, self.pilot_spacing)
        if len(idx) < 2:
            idx = np.array([0, max(n_frames - 1, 1)])
        return idx

    def ls_estimate(self, received_freq, clean_freq, pilots):
        X = clean_freq[pilots]
        Y = received_freq[pilots]
        X_safe = np.where(np.abs(X) < 1e-8, 1e-8 + 0j, X)
        return Y / X_safe  # (n_pilots, frame_len)

    def attention_interpolate(self, H_ls, pilots, n_frames):
        """Attention-style channel tracking: every frame's channel estimate is a
        softmax-weighted blend of the pilot LS estimates, weighted by proximity
        in frame index — a lightweight stand-in for a learned positional
        attention head used in transformer channel estimators."""
        all_idx = np.arange(n_frames)
        dist = np.abs(all_idx[:, None] - pilots[None, :]).astype(float)
        scores = -dist / (self.pilot_spacing / 2.0 + 1e-6)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights @ H_ls  # (n_frames, frame_len)

    def equalize(self, received_freq, H_full):
        H_safe = np.where(np.abs(H_full) < 1e-8, 1e-8 + 0j, H_full)
        return received_freq / H_safe


def voice_metrics(clean, processed):
    mse = np.mean((clean - processed) ** 2)
    power = np.mean(clean ** 2)
    nmse_db = 10 * np.log10(mse / power) if mse > 0 and power > 0 else -100
    return mse, nmse_db


# --- 3. STREAMLIT CONFIG & INITIALIZATION ---
st.set_page_config(layout="wide", page_title="AI-OFDM & Voice Pipeline")

if 'engine' not in st.session_state:
    st.session_state.engine = OFDMTransformerEngine()

if 'data_state' not in st.session_state:
    st.session_state.data_state = "empty"

if 'voice_state' not in st.session_state:
    st.session_state.voice_state = "empty"

eng = st.session_state.engine

# --- 4. UI MAIN LAYOUT ---
st.title("📡 OFDM & 🎙️ Voice Transformer Integration Lab")
st.caption("Advanced DSP Pipeline | Multi-Domain Noise Recovery Engine")

app_mode = st.radio("Select Processing Lab Domain:", ["Wireless OFDM Lab", "Voice DSP Lab"], horizontal=True)

st.markdown("---")

if app_mode == "Wireless OFDM Lab":
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("1. Signal Generation")
        if st.button("🚀 Gen 96x14 Frame", use_container_width=True):
            st.session_state.clean_frame = eng.get_ofdm_frame()
            st.session_state.current_view = st.session_state.clean_frame
            st.session_state.data_state = "clean"

        st.subheader("2. Channel Effects")
        noise_level = st.slider("AWGN Noise Level (σ)", 0.0, 2.0, 0.76)
        if st.button("⛈️ Add Noise & IDFT", use_container_width=True):
            if 'clean_frame' in st.session_state:
                time_domain = eng.apply_idft(st.session_state.clean_frame)
                noisy_time = eng.inject_noise(time_domain, noise_level)
                st.session_state.noisy_frame = eng.apply_dft(noisy_time)
                st.session_state.current_view = st.session_state.noisy_frame
                st.session_state.data_state = "noisy"

        st.subheader("3. Transformer Recovery")
        progress = st.slider("Model Convergence (Alpha)", 0.0, 1.0, 0.5)
        if st.button("🤖 Run Attention Denoising", use_container_width=True):
            if 'noisy_frame' in st.session_state:
                st.session_state.recovered = eng.transformer_denoise(st.session_state.noisy_frame, progress)
                st.session_state.current_view = st.session_state.recovered
                st.session_state.data_state = "recovered"

        if st.button("♻️ Factory Reset", type="primary", use_container_width=True):
            eng.reset()
            st.session_state.data_state = "empty"
            st.rerun()

    with col2:
        if st.session_state.data_state == "empty":
            st.info("Start by generating a frame on the left.")
        else:
            tab1, tab2 = st.tabs(["3D Signal Surface", "IQ Constellation Diagram"])
            view_data = st.session_state.current_view

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Surface(z=view_data[..., 0], colorscale='Blues', name='Real', showscale=False, opacity=0.9))
                fig.add_trace(go.Surface(z=view_data[..., 1], colorscale='Reds', name='Imag', showscale=False, opacity=0.4))
                fig.update_layout(title=f"Stage: {st.session_state.data_state.upper()}",
                                  scene=dict(xaxis_title="Time Slots", yaxis_title="Symbols", zaxis_title="Amp", aspectratio=dict(x=1, y=2, z=0.5)), height=600)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                real_pts = view_data[..., 0].flatten()
                imag_pts = view_data[..., 1].flatten()
                fig_iq = go.Figure()
                fig_iq.add_trace(go.Scatter(x=real_pts, y=imag_pts, mode='markers', marker=dict(color='#81C995', size=4, opacity=0.6)))
                fig_iq.add_trace(go.Scatter(x=[-1, 1, -1, 1], y=[-1, -1, 1, 1], mode='markers', marker=dict(color='red', symbol='x', size=12), name='Ideal'))
                fig_iq.update_layout(title="IQ Constellation", xaxis=dict(range=[-3, 3], title="In-phase"), yaxis=dict(range=[-3, 3], title="Quadrature"), height=600, width=600)
                st.plotly_chart(fig_iq)

        if st.session_state.data_state != "empty" and 'clean_frame' in st.session_state:
            st.markdown("---")
            ber, nmse = eng.calculate_metrics(st.session_state.clean_frame, st.session_state.current_view)
            m1, m2 = st.columns(2)
            m1.metric("Bit Error Rate (BER)", f"{ber:.2f}%")
            m2.metric("NMSE (dB)", f"{nmse:.2f} dB")

else:
    # --- 🎙️ VOICE LAB (now with OFDM-style deep-learning channel estimation) ---
    vcol1, vcol2 = st.columns([1, 2.5])

    with vcol1:
        st.subheader("1. Voice Signal Import")
        voice_source = st.radio("Voice Source:", ["Generate Sample Sine Wave", "Upload .WAV Audio File"])

        if voice_source == "Generate Sample Sine Wave":
            if st.button("🎵 Generate Test Voice Wave", use_container_width=True):
                st.session_state.sample_rate = 16000
                t = np.linspace(0, 1, 16000)
                st.session_state.clean_voice = np.sin(2 * np.pi * 440 * t)
                st.session_state.current_voice_view = st.session_state.clean_voice
                st.session_state.voice_state = "clean"
                for k in ["noisy_voice", "recovered_voice", "H_true", "H_est_full", "received_freq", "equalized_freq"]:
                    st.session_state.pop(k, None)
        else:
            audio_file = st.file_uploader("Upload a standard audio file (.wav)", type=["wav"])
            if audio_file is not None:
                fs, data = wavfile.read(io.BytesIO(audio_file.read()))
                st.session_state.sample_rate = fs

                if len(data.shape) > 1:
                    data = data[:, 0]

                float_data = data.astype(float)
                st.session_state.clean_voice = float_data / np.max(np.abs(float_data))
                st.session_state.current_voice_view = st.session_state.clean_voice
                st.session_state.voice_state = "clean"
                for k in ["noisy_voice", "recovered_voice", "H_true", "H_est_full", "received_freq", "equalized_freq"]:
                    st.session_state.pop(k, None)

        st.subheader("2. OFDM Transmission + Multipath Channel")
        v_noise_level = st.slider("AWGN Noise Intensity", 0.0, 1.5, 0.3)
        frame_len = st.select_slider("OFDM Frame / Subcarrier Length", options=[32, 64, 128, 256], value=128)
        if st.button("📡 Transmit over OFDM + Inject Channel", use_container_width=True):
            if 'clean_voice' in st.session_state:
                estimator = VoiceOFDMChannelEstimator(frame_len=frame_len, pilot_spacing=8)
                clean = st.session_state.clean_voice
                clean_frames = estimator.frame_signal(clean)
                clean_freq = estimator.to_freq(clean_frames)

                taps = estimator.random_channel_taps()
                H_true = estimator.channel_freq_response(taps)

                received_freq = estimator.apply_channel(clean_freq, H_true, v_noise_level)
                received_time = estimator.to_time(received_freq, len(clean))

                st.session_state.estimator = estimator
                st.session_state.clean_freq = clean_freq
                st.session_state.H_true = H_true
                st.session_state.received_freq = received_freq
                st.session_state.original_len = len(clean)

                st.session_state.noisy_voice = received_time
                st.session_state.current_voice_view = received_time
                st.session_state.voice_state = "noisy"
                for k in ["recovered_voice", "H_est_full", "equalized_freq"]:
                    st.session_state.pop(k, None)

        st.subheader("3. Deep-Learning Channel Estimation")
        pilot_spacing = st.slider("Pilot Spacing (fewer pilots = harder estimation)", 2, 32, 8)
        if st.button("🤖 Run Attention-Based Channel Estimation", use_container_width=True):
            if 'received_freq' in st.session_state and 'estimator' in st.session_state:
                estimator = st.session_state.estimator
                estimator.pilot_spacing = pilot_spacing
                received_freq = st.session_state.received_freq
                clean_freq = st.session_state.clean_freq
                n_frames = received_freq.shape[0]

                pilots = estimator.pilot_indices(n_frames)
                H_ls = estimator.ls_estimate(received_freq, clean_freq, pilots)
                H_est_full = estimator.attention_interpolate(H_ls, pilots, n_frames)
                equalized_freq = estimator.equalize(received_freq, H_est_full)
                recovered_time = estimator.to_time(equalized_freq, st.session_state.original_len)

                st.session_state.H_est_full = H_est_full
                st.session_state.equalized_freq = equalized_freq
                st.session_state.pilots = pilots
                st.session_state.recovered_voice = recovered_time
                st.session_state.current_voice_view = recovered_time
                st.session_state.voice_state = "recovered"

    with vcol2:
        if st.session_state.voice_state == "empty":
            st.info("Voice Lab empty. Click 'Generate Test Voice Wave' or upload a file to start.")
        else:
            st.subheader(f"Current Audio State: {st.session_state.voice_state.upper()}")

            sr = st.session_state.get('sample_rate', 16000)
            playback = np.clip(st.session_state.current_voice_view, -1, 1)
            st.audio(playback, sample_rate=sr)

            tabs = st.tabs(["📊 Waveform", "📶 Channel Estimate", "✴️ Frequency Constellation"])

            with tabs[0]:
                fig_voice = go.Figure()
                plot_samples = min(2000, len(st.session_state.current_voice_view))
                color = {'clean': '#4285F4', 'noisy': '#EA4335', 'recovered': '#34A853'}.get(st.session_state.voice_state, '#4285F4')
                fig_voice.add_trace(go.Scatter(
                    y=st.session_state.current_voice_view[:plot_samples],
                    mode='lines', line=dict(color=color), name=st.session_state.voice_state
                ))
                fig_voice.update_layout(xaxis_title="Time Samples", yaxis_title="Amplitude",
                                         yaxis=dict(range=[-3, 3]), height=400)
                st.plotly_chart(fig_voice, use_container_width=True)

            with tabs[1]:
                if 'H_true' in st.session_state:
                    H_true = st.session_state.H_true
                    freqs = np.arange(len(H_true))
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=freqs, y=np.abs(H_true), mode='lines', name='True |H(f)|', line=dict(color='#4285F4')))
                    if 'H_est_full' in st.session_state:
                        H_est_avg = np.mean(np.abs(st.session_state.H_est_full), axis=0)
                        fig_h.add_trace(go.Scatter(x=freqs, y=H_est_avg, mode='lines', name='Estimated |H(f)| (avg over frames)', line=dict(color='#34A853', dash='dash')))
                    fig_h.update_layout(title="Channel Frequency Response: True vs Attention-Estimated",
                                         xaxis_title="Subcarrier Index", yaxis_title="Magnitude", height=380)
                    st.plotly_chart(fig_h, use_container_width=True)

                    if 'H_est_full' in st.session_state:
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=np.abs(st.session_state.H_est_full).T, colorscale='Viridis',
                            colorbar=dict(title="|H_est|")
                        ))
                        fig_heat.update_layout(title="Estimated Channel Magnitude (Subcarrier x Frame)",
                                                xaxis_title="OFDM Frame Index", yaxis_title="Subcarrier Index", height=380)
                        st.plotly_chart(fig_heat, use_container_width=True)
                        st.caption(f"Pilot frames used for LS estimation: {len(st.session_state.get('pilots', []))} "
                                   f"out of {st.session_state.H_est_full.shape[0]} total frames.")
                else:
                    st.info("Run step 2 (Transmit over OFDM) to simulate a channel first.")

            with tabs[2]:
                if 'received_freq' in st.session_state:
                    rx = st.session_state.received_freq.flatten()
                    fig_iq = go.Figure()
                    fig_iq.add_trace(go.Scatter(x=rx.real, y=rx.imag, mode='markers',
                                                 marker=dict(color='#EA4335', size=3, opacity=0.4), name='Received (pre-equalization)'))
                    if 'equalized_freq' in st.session_state:
                        eq = st.session_state.equalized_freq.flatten()
                        fig_iq.add_trace(go.Scatter(x=eq.real, y=eq.imag, mode='markers',
                                                     marker=dict(color='#34A853', size=3, opacity=0.4), name='Equalized (recovered)'))
                    fig_iq.update_layout(title="Frequency-Domain Symbols Before/After Equalization",
                                          xaxis_title="Real", yaxis_title="Imag", height=500)
                    st.plotly_chart(fig_iq, use_container_width=True)
                else:
                    st.info("Run step 2 (Transmit over OFDM) to see frequency-domain symbols.")

            if 'clean_voice' in st.session_state:
                st.markdown("---")
                v_mse, v_nmse = voice_metrics(st.session_state.clean_voice, st.session_state.current_voice_view)
                vm1, vm2 = st.columns(2)
                vm1.metric("Voice Quality Loss Factor", f"{v_mse*100:.2f}%")
                vm2.metric("Audio NMSE (dB)", f"{v_nmse:.2f} dB")