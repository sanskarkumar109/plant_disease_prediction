# 🌿 Plant Disease Detection System

This is a machine learning-based desktop application for real-time plant disease detection and recommendation, using deep learning and AI agents. It is designed to help farmers, gardeners, and researchers identify diseases in plant leaves and provide actionable care advice.

---

## 📌 Features

- 🔍 Detects 11 plant diseases using images of leaves.
- 🤖 Two AI agents:
  - **Agent1: Identifier** – Maps predictions to disease names.
  - **Agent2: Advisor** – Provides disease-specific care tips.
- 🧠 Powered by a MobileNet deep learning model.
- 💻 User-friendly JavaFX frontend interface.
- 🌐 Flask-based backend API for model inference and logic processing.
- 📊 Visual feedback (confidence score, predictions, care info).

---

## 🧰 Tech Stack

| Component         | Technology       |
|------------------|------------------|
| Frontend         | JavaFX (Java)    |
| Backend          | Flask (Python)   |
| Model            | MobileNet (Keras/TensorFlow) |
| Data             | PlantVillage Dataset |
| Communication    | REST API (HTTP)  |

---

## 🚀 How It Works

1. **User uploads an image** of a plant leaf via the JavaFX interface.
2. The image is sent to the **Flask backend**.
3. The **MobileNet model** processes the image and predicts the class.
4. **Agent1** (Identifier) maps the predicted index to a disease name.
5. **Agent2** (Advisor) retrieves care suggestions for that disease from a knowledge base.
6. Results (disease name + advice) are shown in the JavaFX GUI.

---

## 📁 Project Structure

PlantDiseaseDetection/ │ ├── backend/ │ ├── model_training/ │ │ └── mobilenet_planville_model.h5 │ ├── app.py (Flask app) │ ├── agents/ │ │ ├── identifier.py │ │ └── advisor.py │ └── knowledge_base.json │ ├── frontend/ │ └── javafx_ui/ │ └── MainApp.java │ ├── dataset/ │ └── PlantVillage/ │ └── README.md

yaml
Copy
Edit

## 🧪 Testing

To test the backend independently:

```bash
cd backend
python app.py
For JavaFX frontend:

bash
Copy
Edit
# Make sure the Flask backend is running
# Then run the JavaFX project from your IDE (e.g., IntelliJ or Eclipse)
📈 Model Evaluation
Accuracy: 98.5%

Precision / Recall / F1 Score: Computed on a 20% test split of the PlantVillage dataset.

Confusion Matrix & Charts: Available in Project report



