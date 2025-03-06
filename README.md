# Signetix - Hand Sign Recognition

Signetix is a **real-time hand gesture recognition system** that converts hand signs into **text and speech**. 
It utilizes **computer vision** and **deep learning** techniques to recognize predefined gestures.

---

## 🚀 Features
- 🎥 **Live Gesture Recognition** – Uses a webcam to recognize hand signs in real-time.
- 🗣️ **Speech Output** – Converts recognized gestures into spoken words using `gTTS`.
- 📷 **Image Upload Support** – Allows gesture recognition from uploaded images.
- 🎨 **Customizable UI** – Includes gradient shadow effects and a voice feedback toggle.
- 🏗️ **Modular Design** – Easily extendable and well-structured for future improvements.

---

## 📂 Folder Structure
```
Signetix/
│── app.py                # Main Streamlit application
│── requirements.txt       # Dependencies
│── model/                
│   ├── keras_model.h5     # Trained deep learning model
│   ├── labels.txt         # Gesture label mappings
│── data_collection/       
│   ├── data_collection.py # Script for collecting hand sign data
│── README.md              # Project documentation
```

---

## 🛠️ Setup & Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/Anqwert/Signetix.git
cd Signetix
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Application Locally**
```bash
streamlit run app.py
```

---

## 🎨 Customization Options

### 🔹 **Modify Gesture Labels**
- Open `model/labels.txt` and update the labels.
- Ensure the order matches the model’s training.

### 🔹 **Change UI Appearance**
- Update colors, fonts, or effects inside `app.py`.
- Example: Modify the gradient effect in the bounding box for a different color scheme.

### 🔹 **Enable/Disable Voice Feedback**
- Toggle voice feedback from the Streamlit sidebar.

### 🔹 **Train with More Gestures**
- Use `data_collection/data_collection.py` to collect more hand gesture images.
- Train a new model and replace `model/keras_model.h5`.

---

## 🌟 Future Enhancements
- 🔍 **Support for more gestures**.
- 🎙️ **Offline speech synthesis using pyttsx3**.
- 📈 **Improved model accuracy with more training data**.
- 📊 **Analytics for gesture recognition performance**.

---

## 📜 License
This project is **open-source** and free to use.

---

## 🤝 Contributing
Want to improve Signetix? Feel free to fork the repository, make enhancements, and submit a pull request.

---

Made with ❤️ by **Anqwert**  
