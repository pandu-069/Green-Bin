# 💡 Waste Scanner Web App

An intelligent Streamlit-based waste classification web app that allows users to upload an image and enter a related query. It then uses two YOLOv8 models to detect objects and generates brief insights about recyclability and environmental impact.

## 🔥 Features
- Upload an image and view it in the app
- Enter a prompt related to the uploaded image
- Detects objects using **two YOLOv8 models**:
  - 🔬 A **custom-trained YOLOv8 model** trained with a vast dataset specifically for garbage and waste classification
  - 🌍 The general-purpose **YOLOv8n model** for additional object detection
- Combines detection results with user input to generate smart insights
- Displays detection results side-by-side for easy comparison
- Beautiful, interactive Streamlit UI

## 🧠 Powered By
- **Custom YOLOv8 Model** (garbage_classifier_enhanced)
- **YOLOv8n** (Ultralytics)
- **Streamlit** for the user interface

## 🛠️ Installation

1. Clone the repository or download the files.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📦 requirements.txt
```
streamlit>=1.32.0
openai>=1.10.0
Pillow
```

## 🧪 Usage
- Open your browser to `http://localhost:8501`
- Upload a waste image (jpg, png, etc.)
- Type your query (e.g., "Is this item recyclable? What harm can it cause?")
- View detected objects from both models and read generated insights

## 📌 Example Prompt
> "This image shows plastic wrappers and a broken mobile phone. Are these recyclable? What impact do they have on the environment?"

## 📄 License
MIT License. Free for educational and personal use.

---
Made with ❤️ using Custom YOLOv8 + YOLOv8n

