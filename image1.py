import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="📸 Photo Enhancement App", layout="wide")
st.title("📸 Photo Enhancement App")

# ===== File Upload / Camera =====
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("Use Camera to Capture Image")
if use_camera:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        uploaded_file = camera_image

# ===== Processing Functions =====
def your_method(img):
    """Your original image processing steps from notebook."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def enhanced_method(img, denoise_strength, alpha, beta):
    """My enhancement method: denoise + contrast + histogram equalization."""
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)
    img_contrast = cv2.convertScaleAbs(img_denoised, alpha=alpha, beta=beta)
    img_yuv = cv2.cvtColor(img_contrast, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_eq

# ===== Main App =====
if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Adjust Enhancement Settings")
    denoise_strength = st.slider("Denoising Strength", 0, 30, 10)
    alpha = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0)
    beta = st.slider("Brightness (beta)", -100, 100, 0)

    # Process images
    result_your = your_method(img)
    result_enhanced = enhanced_method(img, denoise_strength, alpha, beta)

    # Display side-by-side
    st.subheader("Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original", channels="RGB")
    with col2:
        st.image(result_your, caption="Your Method", channels="RGB")
    with col3:
        st.image(result_enhanced, caption="Enhanced Method", channels="RGB")

    # Download buttons
    st.subheader("Download Results")
    def convert_img(img_array):
        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return buf.getvalue()

    st.download_button("⬇ Download Your Method Result", convert_img(result_your), "your_method.png", "image/png")
    st.download_button("⬇ Download Enhanced Result", convert_img(result_enhanced), "enhanced_method.png", "image/png")
else:
    st.info("📤 Upload an image or use your camera to start.")
