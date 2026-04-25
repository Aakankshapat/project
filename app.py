import streamlit as st
import pickle
from PIL import Image, ImageFilter
import numpy as np
import re
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# ----
# Load Text Model
# ----
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(BASE_DIR / "text_model5.pkl", "rb"))
        vectorizer = pickle.load(open(BASE_DIR / "text_vectorizer5.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return model, vectorizer

text_model, text_vectorizer = load_model()

# ----
# Text Cleaning Function
# ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_ocr_image(image: Image.Image):
    if CV2_AVAILABLE:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img, 3)
        return img
    return image.convert("L").filter(ImageFilter.MedianFilter(size=3))

# ----
# Page Config
# ----
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ----
# UI Styling
# ----
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
}
</style>
""", unsafe_allow_html=True)

# ----
# Title
# ----
st.markdown('<div class="title">📰 Fake News Detection System</div>', unsafe_allow_html=True)

if TESSERACT_AVAILABLE:
    st.markdown('<div class="subtitle">Detect Fake or Real News using Text & Images</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="subtitle">Detect Fake or Real News using Text</div>', unsafe_allow_html=True)
    st.warning("⚠️ Image OCR not available. Please install Tesseract OCR for image support.")

st.write("")

# ----
# Mode Selection
# ----
modes = ["Text News"]
if TESSERACT_AVAILABLE:
    modes.append("Image News")

option = st.radio("Choose Input Type:", modes, horizontal=True)

if option == "Text News":
    st.subheader("📄 Paste News Text")
    user_text = st.text_area("Enter News Content")
    if st.button("Detect Text News"):
        if user_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            cleaned = clean_text(user_text)
            vector = text_vectorizer.transform([cleaned])
            prediction = text_model.predict(vector)[0]
            if prediction == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

elif option == "Image News" and TESSERACT_AVAILABLE:
    if not CV2_AVAILABLE:
        st.info("OpenCV not installed; OCR will use PIL fallback.")
    st.subheader("🖼 Upload News Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Detect Image News"):
            try:
                ocr_image = prepare_ocr_image(image)
                extracted_text = pytesseract.image_to_string(ocr_image)
                st.subheader("📝 Extracted Text")
                st.write(extracted_text if extracted_text.strip() else "No text detected")
                cleaned = clean_text(extracted_text)
                vector = text_vectorizer.transform([cleaned])
                prediction = text_model.predict(vector)[0]
                if prediction == 1:
                    st.success("✅ REAL NEWS IMAGE")
                else:
                    st.error("❌ FAKE NEWS IMAGE")
            except Exception as e:
                st.error(f"Error processing image: {e}")

st.write("---")
st.write("Built with ❤️ using Streamlit")
