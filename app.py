import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="✏️",
    layout="centered"
)

# ── Model (identical to training definition) ─────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(
        torch.load("model/mnist_cnn.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()

# ── Preprocessing ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def preprocess(img_array: np.ndarray) -> torch.Tensor:
    """
    Canvas gives RGBA numpy array, white background, dark strokes.
    MNIST expects: white digit on black background.
    Steps: RGBA → greyscale → invert → normalise
    """
    img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
    img = Image.fromarray(255 - np.array(img))   # invert: dark strokes → white digit
    return transform(img).unsqueeze(0)            # (1, 1, 28, 28)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("✏️ Digit Classifier")
st.write("Draw a digit (0–9) in the box below and hit **Predict**.")

# Drawing canvas — white bg, black brush, matches MNIST after inversion
canvas_result = st_canvas(
    fill_color    = "rgba(0,0,0,0)",
    stroke_width  = 20,
    stroke_color  = "#000000",
    background_color = "#FFFFFF",
    width         = 280,
    height        = 280,
    drawing_mode  = "freedraw",
    key           = "canvas",
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

with col2:
    # Rerun with a fresh key to clear the canvas
    if st.button("🗑 Clear", use_container_width=True):
        st.rerun()

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if canvas_result.image_data is None:
        st.warning("Draw something first!")
    else:
        # Check canvas isn't blank (all-white)
        img_array = canvas_result.image_data
        if img_array[:, :, :3].mean() > 250:
            st.warning("The canvas looks empty — please draw a digit first.")
        else:
            with torch.no_grad():
                tensor = preprocess(img_array)
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                conf, pred = torch.max(probs, 0)

            digit      = int(pred.item())
            confidence = float(conf.item())

            # ── Results display ───────────────────────────────────────────
            st.divider()

            c1, c2 = st.columns([1, 2])

            with c1:
                st.metric(label="Predicted digit", value=str(digit))
                st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

            with c2:
                st.write("**Probability for each digit**")
                prob_data = {str(i): float(probs[i]) for i in range(10)}
                st.bar_chart(prob_data, height=200)

            # ── What the model saw ────────────────────────────────────────
            with st.expander("What the model actually saw (28×28)"):
                img_array = canvas_result.image_data
                img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
                img_inv = Image.fromarray(255 - np.array(img))
                img_small = img_inv.resize((28, 28), Image.LANCZOS)
                img_display = img_small.resize((140, 140), Image.NEAREST)  # upscale for visibility
                st.image(img_display, caption="Resized to 28×28 (what the CNN receives)")