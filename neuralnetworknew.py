
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time


# --- 1. THE ADVANCED ENGINE ---
class ComplexNeuralNetwork:
    def __init__(self, arch):
        self.arch = arch
        self.reset()

    def reset(self):
        # Mathematical Zero Reset
        self.weights = [np.zeros((self.arch[i], self.arch[i + 1])) for i in range(len(self.arch) - 1)]
        self.biases = [np.zeros(self.arch[i + 1]) for i in range(len(self.arch) - 1)]
        self.loss_history = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def train_complex_step(self, x, target_func_val, lr):
        # Break symmetry across all layers with proper initialization
        if all(np.all(w == 0) for w in self.weights):
            for i in range(len(self.weights)):
                # Xavier/Glorot initialization for better training
                self.weights[i] = np.random.randn(*self.weights[i].shape) * np.sqrt(2.0 / self.arch[i])
                self.biases[i] = np.zeros(self.biases[i].shape)  # Start biases at zero

        # Forward Pass (fixed - linear output for regression)
        acts = [x]
        z_values = []  # Store pre-activation values for backprop

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(acts[-1], w) + b
            z_values.append(z)

            if i < len(self.weights) - 1:  # Hidden layers use ReLU
                a = self.relu(z)
            else:  # Output layer - linear activation for regression
                a = z
            acts.append(a)

        # Calculate loss (MSE)
        error = acts[-1] - target_func_val
        self.loss_history.append(np.mean(error ** 2))

        # Backpropagation (fixed)
        # For output layer with linear activation, delta = error
        delta = error

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Reshape delta if needed for proper gradient calculation
            if delta.ndim == 0:
                delta = np.array([delta])
            if acts[i].ndim == 0:
                acts_i = np.array([acts[i]])
            else:
                acts_i = acts[i]

            # Calculate gradients
            grad_w = np.outer(acts_i, delta)
            grad_b = delta

            # Update weights and biases
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

            # Propagate delta to previous layer (if not input layer)
            if i > 0:
                # Reshape for proper matrix multiplication
                if delta.ndim == 1:
                    delta = delta.reshape(1, -1)
                if self.weights[i].ndim == 2:
                    # delta_prev = (delta * W^T) * relu_deriv(z_prev)
                    delta = np.dot(delta, self.weights[i].T).flatten()
                    delta = delta * self.relu_deriv(z_values[i - 1])

        return acts


# --- 2. SESSION STATE ---
if 'brain' not in st.session_state:
    st.session_state.brain = ComplexNeuralNetwork([3, 8, 8, 6, 4, 1])  # Adjusted architecture
if 'run_flow' not in st.session_state:
    st.session_state.run_flow = False
if 'loss_values' not in st.session_state:
    st.session_state.loss_values = []

brain = st.session_state.brain

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide")
st.title("DL Flow Visualization")

col1, col2 = st.columns([1, 3.5])

with col1:
    st.subheader("Variable Inputs")
    x1 = st.slider("Input X1", -2.0, 2.0, 1.0, step=0.1)
    x2 = st.slider("Input X2", -2.0, 2.0, 1.0, step=0.1)
    x3 = st.slider("Input X3", -2.0, 2.0, 1.0, step=0.1)

    inputs = np.array([x1, x2, x3])

    # COMPLEX FORMULA (non-linear target)
    complex_target = (x1 ** 2) + (np.sin(x2) * x3) + (x1 * x2 * 0.5)
    st.info(
        f"**Target Formula:**\n$y = x_1^2 + \\sin(x_2) \\cdot x_3 + 0.5 \\cdot x_1 \\cdot x_2$\n\n**Current Target Value:** {complex_target:.4f}")

    # Training controls
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Visualize Flow", use_container_width=True):
            st.session_state.run_flow = True
    with col_b:
        if st.button("🔄 Train One Step", use_container_width=True):
            # Train without visualization
            final_acts = brain.train_complex_step(inputs, np.array([complex_target]), lr=0.01)
            st.session_state.loss_values = brain.loss_history[-10:]  # Store last 10 loss values
            st.rerun()

    st.divider()
    if st.button("♻️ Reset Network", type="primary", use_container_width=True):
        brain.reset()
        st.session_state.loss_values = []
        st.rerun()

    # Show loss history
    if st.session_state.loss_values:
        st.subheader("Recent Loss Values")
        loss_df = [f"{i + 1}: {loss:.6f}" for i, loss in enumerate(st.session_state.loss_values)]
        st.code("\n".join(loss_df))

# --- 4. FLOW VISUALIZATION ---
plot_area = col2.empty()
info_area = col2.empty()
prediction_area = col2.empty()


def draw_nn(activations, active_idx=None):
    fig = go.Figure()
    palette = {'in': '#AECBFA', 'hid': '#FEEFC3', 'out': '#81C995', 'active': '#FF7043'}
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

    for i, layer_vals in enumerate(activations):
        # Handle both scalar and array values
        if np.isscalar(layer_vals):
            layer_vals = [layer_vals]
        else:
            layer_vals = layer_vals.flatten() if layer_vals.ndim > 1 else layer_vals

        n = len(layer_vals)
        y_pos = np.linspace(n / 2, -n / 2, n) if n > 1 else [0]

        for j, val in enumerate(layer_vals):
            node_x.append(i * 1.8)  # Increased spacing
            node_y.append(y_pos[j])
            node_text.append(f"{float(val):.3f}")

            # Color coding
            if i == active_idx:
                node_color.append(palette['active'])
                node_size.append(55)
            elif i == 0:
                node_color.append(palette['in'])
                node_size.append(45)
            elif i == len(activations) - 1:
                node_color.append(palette['out'])
                node_size.append(45)
            else:
                node_color.append(palette['hid'])
                node_size.append(42)

    # Draw edges
    edge_x, edge_y = [], []
    for i in range(len(brain.arch) - 1):
        n_source = brain.arch[i]
        n_target = brain.arch[i + 1]

        y_source = np.linspace(n_source / 2, -n_source / 2, n_source) if n_source > 1 else [0]
        y_target = np.linspace(n_target / 2, -n_target / 2, n_target) if n_target > 1 else [0]

        for y1 in y_source:
            for y2 in y_target:
                edge_x.extend([i * 1.8, (i + 1) * 1.8, None])
                edge_y.extend([y1, y2, None])

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color='#CCCCCC'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='white'),
            symbol='circle'
        )
    ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=600,
        width=900,
        xaxis=dict(visible=False, range=[-0.5, len(brain.arch) * 1.8 - 0.5]),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# Animation logic
if st.session_state.get('run_flow', False):
    # Train the network
    with st.spinner('Computing neural network flow...'):
        final_acts = brain.train_complex_step(inputs, np.array([complex_target]), lr=0.02)
        st.session_state.loss_values = brain.loss_history[-10:]

    # Show forward pass animation
    current_show = [inputs]

    # Input layer display
    plot_area.plotly_chart(draw_nn(current_show, active_idx=0), use_container_width=True)
    info_area.markdown("**Layer 1 (Input):** Raw input values")
    time.sleep(0.8)

    # Hidden layers animation
    for layer_idx in range(len(brain.weights)):
        # Calculate next layer
        w = brain.weights[layer_idx]
        b = brain.biases[layer_idx]
        z = np.dot(current_show[-1], w) + b

        if layer_idx < len(brain.weights) - 1:
            a = brain.relu(z)
            activation_name = "ReLU"
        else:
            a = z  # Linear output
            activation_name = "Linear"

        current_show.append(a)

        # Update visualization
        plot_area.plotly_chart(draw_nn(current_show, active_idx=layer_idx + 1), use_container_width=True)

        # Show calculation info
        if layer_idx < len(brain.weights) - 1:
            info_area.latex(
                rf"Layer\ {layer_idx + 2}\ (Hidden):\ a_{layer_idx + 2} = \max(0, W_{layer_idx + 1} \cdot a_{layer_idx + 1} + b_{layer_idx + 1})"
            )
        else:
            info_area.latex(
                rf"Layer\ {layer_idx + 2}\ (Output):\ \hat{{y}} = W_{layer_idx + 1} \cdot a_{layer_idx + 1} + b_{layer_idx + 1}"
            )
        time.sleep(0.7)

    # Show final prediction
    prediction = final_acts[-1][0] if hasattr(final_acts[-1], '__len__') else final_acts[-1]
    error = abs(prediction - complex_target)

    if error < 0.1:
        result_emoji = "✅ Excellent!"
    elif error < 0.5:
        result_emoji = "👍 Good"
    elif error < 1.0:
        result_emoji = "🔄 Learning"
    else:
        result_emoji = "📚 Training"

    prediction_area.success(
        f"{result_emoji} Network Prediction: **{prediction:.4f}**  \n"
        f"Target Value: **{complex_target:.4f}**  \n"
        f"Error: **{error:.4f}**"
    )

    st.session_state.run_flow = False

else:
    # Static display
    current_show = [inputs]
    for layer_idx in range(len(brain.weights)):
        if layer_idx < len(current_show) - 1:  # Skip if already calculated
            w = brain.weights[layer_idx]
            b = brain.biases[layer_idx]
            z = np.dot(current_show[-1], w) + b
            if layer_idx < len(brain.weights) - 1:
                a = brain.relu(z)
            else:
                a = z
            current_show.append(a)

    plot_area.plotly_chart(draw_nn(current_show), use_container_width=True)

    # Show current prediction if available
    if len(current_show) > len(brain.weights):
        current_pred = current_show[-1][0] if hasattr(current_show[-1], '__len__') else current_show[-1]
        prediction_area.info(f"Current Network Output: **{current_pred:.4f}**  \nTarget: **{complex_target:.4f}**")
    else:
        prediction_area.info("Click 'Visualize Flow' to see the network in action!")

# Add footer with explanation
st.markdown("---")
# Add footer with explanation

st.markdown("""
**How it works:**
- **Input Layer (3 neurons)**: Takes X₁, X₂, X₃ values
- **Hidden Layers**: Apply ReLU activation: a = max(0, W·x + b)
- **Output Layer (1 neuron)**: Linear activation for regression
- **Target Function**: y = x₁² + sin(x₂)·x₃ + 0.5·x₁·x₂

The network learns to approximate this non-linear function through training!
""")
st.markdown("---")
