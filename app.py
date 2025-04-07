import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import openai

# 🔐 Set your OpenAI API key
openai.api_key = "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"

# Load YOLO models
model1 = YOLO(r"garbage_classifier_enhanced\runs\detect\garbage_classifier_enhanced\weights\best.pt")
model2 = YOLO("yolov8n.pt")

# Streamlit page settings
st.set_page_config(page_title="♻️ Smart Waste Scanner", layout="wide")
st.markdown("## ♻️ Smart Waste Scanner using YOLOv8 + GPT")
st.markdown("Upload an image of waste, give a context prompt, and get AI-powered recycling info.")

# --- UI Input Section ---
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
with col2:
    user_prompt = st.text_input("🧠 Enter a short context or prompt (e.g., 'This was found near my house')", max_chars=100)

# --- Processing ---
if uploaded_file and user_prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting items and fetching info from ChatGPT..."):
        # Save temp image for YOLO
        img_np = np.array(image)
        temp_path = "temp_uploaded.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        detected_labels = []

        # Garbage model detection
        results1 = model1.predict(temp_path)
        img1 = results1[0].plot()
        for box in results1[0].boxes:
            cls_id = int(box.cls.item())
            label = model1.names[cls_id]
            detected_labels.append(label)

        # YOLOv8 general detection
        results2 = model2.predict(temp_path)
        img2 = results2[0].plot()
        for box in results2[0].boxes:
            cls_id = int(box.cls.item())
            label = model2.names[cls_id]
            detected_labels.append(label)

        # Show detection results
        col3, col4 = st.columns(2)
        with col3:
            st.image(img1, caption="🗑️ Garbage Classifier", use_column_width=True)
        with col4:
            st.image(img2, caption="🔍 YOLOv8 General Detector", use_column_width=True)

        # Show unique labels
        unique_labels = list(set(detected_labels))
        st.markdown("### ✅ Detected Labels")
        st.success(", ".join(unique_labels) if unique_labels else "No objects detected.")

        # --- GPT Info Section ---
        gpt_input = (
            f"These items were detected: {', '.join(unique_labels)}. "
            f"User context: {user_prompt}. "
            "Please give a concise 2-3 sentence summary on whether these are recyclable, "
            "their impact on the environment, and any safe disposal tips. Keep the tone informative and friendly. Give More preferece to the user context while considering detected items"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": gpt_input}],
                max_tokens=150,
                temperature=0.7,
            )
            reply = response["choices"][0]["message"]["content"]

            st.markdown("### 🌿 Environmental Insight from AI")
            st.info(reply)

        except Exception as e:
            st.error(f"Failed to connect to GPT API: {str(e)}")

        os.remove(temp_path)
