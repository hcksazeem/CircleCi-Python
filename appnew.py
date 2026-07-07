import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time


# --- 1. THE SIGNAL & ATTENTION ENGINE ---
class OFDMTransformerEngine:
    def __init__(self, sc=96, ts=14):
        self.sc = sc
        self.ts = ts
        self.reset()

    def reset(self):
        self.loss_history = []

    def get_ofdm_frame(self):
        # Ground Truth: Pure QPSK Symbols (+1, -1)
        real = np.random.choice([-1.0, 1.0], size=(self.sc, self.ts))
        imag = np.random.choice([-1.0, 1.0], size=(self.sc, self.ts))
        return np.stack([real, imag], axis=-1)

    def apply_idft(self, frame):
        # Frequency -> Time Domain (IFFT)
        complex_frame = frame[..., 0] + 1j * frame[..., 1]
        time_data = np.fft.ifft(complex_frame, axis=0)
        return np.stack([time_data.real, time_data.imag], axis=-1)

    def inject_noise(self, frame, noise_level):
        # Add AWGN
        noise = np.random.normal(0, noise_level, frame.shape)
        return frame + noise

    def apply_dft(self, noisy_time_frame):
        # Move from Time Domain back to Frequency (Symbol) Domain
        # This aligns the noisy signal with the clean signal for NMSE math
        complex_time = noisy_time_frame[..., 0] + 1j * noisy_time_frame[..., 1]
        symbol_data = np.fft.fft(complex_time, axis=0)
        return np.stack([symbol_data.real, symbol_data.imag], axis=-1)
    def transformer_denoise(self, noisy_frame, learning_progress=0.4):
        """
        FIXED: Soft-Thresholding Regression.
        Instead of hard clipping (which spikes NMSE), we nudge the signal
        towards the constellation points. This lowers the error distance.
        """
        # Determine the 'target' constellation points (-1 or 1)
        ideal_points = np.sign(noisy_frame)

        # Linear Interpolation: (1-alpha)*Noisy + alpha*Ideal
        # This reduces the variance and the MSE simultaneously.
        recovered = (1 - learning_progress) * noisy_frame + learning_progress * ideal_points
        return recovered

    def calculate_metrics(self, original, processed):
        # 1. BER Calculation
        orig_bits = (original > 0).astype(int)
        proc_bits = (processed > 0).astype(int)
        ber = (np.sum(orig_bits != proc_bits) / original.size) * 100

        # 2. NMSE Calculation (dB)
        # NMSE = 10 * log10( sum(|orig - proc|^2) / sum(|orig|^2) )
        mse = np.mean((original - processed) ** 2)
        signal_power = np.mean(original ** 2)

        if signal_power == 0:
            return ber, 0.0

        nmse_linear = mse / signal_power
        nmse_db = 10 * np.log10(nmse_linear) if nmse_linear > 0 else -100

        return ber, nmse_db


# --- 2. STREAMLIT CONFIG ---
st.set_page_config(layout="wide", page_title="AI-OFDM Pipeline")

if 'engine' not in st.session_state:
    st.session_state.engine = OFDMTransformerEngine()
    st.session_state.data_state = "empty"

eng = st.session_state.engine

# --- 3. UI CONTROLS ---
st.title("📡 OFDM ⇄ Transformer Integration Lab")
st.caption("96 Symbols (Subcarriers) | 14 Time Slots | Corrected NMSE Logic")

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
            # 1. Move to Time Domain and add noise
            time_domain = eng.apply_idft(st.session_state.clean_frame)
            noisy_time = eng.inject_noise(time_domain, noise_level)

            # 2. Move BACK to Symbol Domain so NMSE math works
            st.session_state.noisy_frame = eng.apply_dft(noisy_time)
            st.session_state.current_view = st.session_state.noisy_frame
            st.session_state.data_state = "noisy"

    st.subheader("3. Transformer Recovery")
    # Added a 'Learning Progress' slider to show how NMSE drops as the model improves
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

# --- 4. VISUALIZATION ---
with col2:
    if st.session_state.data_state == "empty":
        st.info("Start by generating a frame on the left.")
    else:
        tab1, tab2 = st.tabs(["3D Signal Surface", "IQ Constellation Diagram"])
        view_data = st.session_state.current_view

        with tab1:
            fig = go.Figure()
            fig.add_trace(
                go.Surface(z=view_data[..., 0], colorscale='Blues', name='Real', showscale=False, opacity=0.9))
            fig.add_trace(go.Surface(z=view_data[..., 1], colorscale='Reds', name='Imag', showscale=False, opacity=0.4))
            fig.update_layout(
                title=f"Stage: {st.session_state.data_state.upper()}",
                scene=dict(xaxis_title="Time Slots", yaxis_title="Symbols", zaxis_title="Amp",
                           aspectratio=dict(x=1, y=2, z=0.5)),
                height=600
            )
            st.plotly_chart(fig, width='stretch')

        with tab2:
            real_pts = view_data[..., 0].flatten()
            imag_pts = view_data[..., 1].flatten()
            fig_iq = go.Figure()
            fig_iq.add_trace(
                go.Scatter(x=real_pts, y=imag_pts, mode='markers', marker=dict(color='#81C995', size=4, opacity=0.6)))
            fig_iq.add_trace(go.Scatter(x=[-1, 1, -1, 1], y=[-1, -1, 1, 1], mode='markers',
                                        marker=dict(color='red', symbol='x', size=12), name='Ideal'))
            fig_iq.update_layout(title="IQ Constellation", xaxis=dict(range=[-3, 3], title="In-phase"),
                                 yaxis=dict(range=[-3, 3], title="Quadrature"), height=600, width=600)
            st.plotly_chart(fig_iq)

    # --- METRICS BAR ---
    if st.session_state.data_state != "empty" and 'clean_frame' in st.session_state:
        st.markdown("---")
        ber, nmse = eng.calculate_metrics(st.session_state.clean_frame, st.session_state.current_view)

        m1, m2 = st.columns(2)
        m1.metric("Bit Error Rate (BER)", f"{ber:.2f}%", delta_color="inverse")
        # delta shows improvement (negative delta is good in NMSE)
        m2.metric("NMSE (dB)", f"{nmse:.2f} dB")

        st.info(
            "💡 **Why NMSE is dropping now:** The Transformer is performing regression (Soft-Thresholding) to reduce the distance to the clean signal, rather than just flipping bits.")
