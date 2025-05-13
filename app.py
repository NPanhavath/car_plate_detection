from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import cv2
import easyocr

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = YOLO("runs/detect/train6/weights/best.pt")
reader = easyocr.Reader(['en'])

@app.route("/", methods=["GET", "POST"])
def index():
    plate_text = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Predict with YOLO
            results = model.predict(filepath, conf=0.25)
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # OCR on plate area
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = image[y1:y2, x1:x2]
                    text = reader.readtext(crop, detail=0)
                    plate_text = " ".join(text).strip()

            image_url = f"uploads/{file.filename}"

    return render_template("index.html", plate=plate_text, image_url=image_url)
if __name__ == "__main__":
    app.run(debug=True)

