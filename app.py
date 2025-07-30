import streamlit as st
import face_recognition
import numpy as np
import os
from PIL import Image
import cv2

# পরিচিত মুখগুলো লোড করি
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(os.path.splitext(filename)[0])

st.title("🧠 Face Recognition App")

uploaded_file = st.file_uploader("📸 একটা ছবি আপলোড করো", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # আপলোড করা ছবি দেখাও
    image = Image.open(uploaded_file)
    st.image(image, caption="আপলোড করা ছবি", use_column_width=True)

    # OpenCV ফরম্যাটে রূপান্তর
    image_np = np.array(image)
    rgb_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # মুখ খোঁজো
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # ফেস এর চারপাশে বক্স ও নাম
        cv2.rectangle(rgb_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    st.image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), caption="Face Detection Result", use_column_width=True)
