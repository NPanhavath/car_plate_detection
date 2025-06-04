from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
import easyocr
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 trained model
model = YOLO("runs/detect/train/weights/best.pt")
reader = easyocr.Reader(['en'])

@app.route("/", methods=["GET", "POST"])
def index():
    plate_text = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Predict with YOLOv8
            results = model.predict(filepath, conf=0.25)
            image = cv2.imread(filepath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    crop = image_rgb[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (300, 100))

                    # Preprocess for better OCR
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(blurred)

                    # OCR with allowlist
                    text = reader.readtext(
                        enhanced,
                        detail=0,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                    plate_text = " ".join(text).strip()

                    # Draw results on image
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_rgb, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 0), 2)

            # Save final output image
            result_path = os.path.join(UPLOAD_FOLDER, "result_" + filename)
            cv2.imwrite(result_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            image_url = f"uploads/result_{filename}"

    return render_template("index.html", plate=plate_text, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
