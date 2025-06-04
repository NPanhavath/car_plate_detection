# 🚗 Car Plate Detection Web App

This project is a Flask-based web application that detects car license plates from images using a pre-trained YOLO model and extracts the text using EasyOCR.

---

## 📁 Project Structure

```
car_plate_detection/
├── static/
├── templates/
├── dataset/
├── cp_detect_clean.ipynb
├── app.py
└── requirements.txt
```

---

## 💻 How to Run This Project

### 🔽 1. Clone or Download the Repo

```bash
git clone https://github.com/NPanhavath/car_plate_detection.git
cd car_plate_detection
```

Or just click **Code > Download ZIP** and extract it.

---

### 🐍 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🏁 4. Run the Web App

```bash
python app.py
```

Then open your browser and go to:  
👉 `http://127.0.0.1:5000/`

---

### 🧠 Features
- Upload image of a car
- Detect license plate using YOLO
- Extract text from plate using EasyOCR
- Show results in browser

---

## ⚠️ Notes
- Make sure `best.pt` is in the correct YOLO path (inside `runs/detect/train/weights/`)
- Internet is needed to load EasyOCR language model the first time
- For larger files or real-time video, you may need additional setup

---

## 📄 License

This project is for educational and research use only.
