import streamlit as st
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace

st.set_page_config(page_title="Face Analyzer", layout="centered")

st.title("🔍 AI Face Analyzer (Age, Gender, Emotion)")
st.write("আপলোড করা ছবিতে মুখ বিশ্লেষণ করে বয়স, লিঙ্গ, আবেগ বের করা হয়।")

uploaded_file = st.file_uploader("📤 একটা ছবি আপলোড করো", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="আপলোড করা ছবি", use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Analyze using DeepFace
    with st.spinner("Face বিশ্লেষণ হচ্ছে..."):
        try:
            results = DeepFace.analyze(img_bgr, actions=["age", "gender", "emotion"], enforce_detection=False)
            result = results[0]

            st.subheader("📊 বিশ্লেষণ ফলাফল:")
            st.markdown(f"""
            - 👤 **Age**: {result['age']}  
            - 🚻 **Gender**: {result['gender']}  
            - 😊 **Emotion**: {result['dominant_emotion']}
            """)
        except Exception as e:
            st.error(f"❌ বিশ্লেষণ ব্যর্থ হয়েছে: {e}")
