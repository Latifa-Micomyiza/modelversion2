import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import os
import re

# Load YOLOv8 model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Required ID features & confidence thresholds
required_classes = {
    "coat of arms": 0.70,
    "rwandan flag": 0.70,
    "ID number": 0.80,
    "hologram":0.70,
}

def preprocess_image(cropped_image):
    """Preprocess image for OCR: convert to grayscale, blur, thresholding, and resize."""
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return resized

def extract_text(image, box):
    """Extract text from a given bounding box using pytesseract."""
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    preprocessed_image = preprocess_image(cropped_image)
    text = pytesseract.image_to_string(preprocessed_image, config="--psm 3")
    return text.strip()

def is_rwandan_id(results):
    """Check if the image contains all required features for a Rwandan ID."""
    detected_classes = set()
    detected_classes_info = []

    if not results:
        print("No detections found.")
        return False, detected_classes_info

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            print(f"Class: {class_name}, Confidence: {confidence}")

            # Add to detected_classes if the class is in required_classes and meets the threshold
            if class_name in required_classes:
                print(f"Class '{class_name}' is in required_classes with threshold {required_classes[class_name]}")
                if confidence >= required_classes[class_name]:
                    detected_classes.add(class_name)
                    print(f"Added '{class_name}' to detected_classes")

            # Add to detected_classes_info if confidence >= 0.70
            if confidence >= 0.70:
                detected_classes_info.append({"class_name": class_name, "confidence": confidence})
                print(f"Appended '{class_name}' to detected_classes_info")

    print(f"Detected Classes: {detected_classes}")
    print(f"Required Classes: {set(required_classes.keys())}")

    return all(cls in detected_classes for cls in required_classes.keys()), detected_classes_info
def process_image(file_path):
    """Process the uploaded image to detect features and extract text."""
    image = cv2.imread(file_path)
    if image is None:
        return {"error": "Failed to process image"}

    results = model(image)

    id_valid, detected_classes_info = is_rwandan_id(results)

    extracted_details = {
        "Names": None,
        "Date of birth": None,
        "Place of issue": None,
        "Gender": None,
        "ID number": None,
    }

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)

            if class_name in extracted_details and confidence >= required_classes.get(class_name, 0):
               extracted_details[class_name] = extract_text(image, box.xyxy[0])

    print("Extracted Details:", extracted_details)  # Debugging output

    formatted_details = [
        {"key": "namesOnNationalId", "name": "Amazina / Names", "value": extracted_details["Names"]},
        {"key": "dateOfBirth", "name": "Itariki yavutseho / Date of Birth", "value": extracted_details["Date of birth"]},
        {"key": "placeOfIssue", "name": "Aho Yatangiwe / Place of Issue", "value": extracted_details["Place of issue"]},
        {"key": "gender", "name": "Igitsina / Sex", "value": extracted_details["Gender"]},
        {"key": "nationalId", "name": "Indangamuntu / National ID No", "value": extracted_details["ID number"]}
    ]

    os.remove(file_path)

    final_output = {
        "verified": id_valid,
        "documentDetails": formatted_details,
        "message": "ID successfully authenticated as Rwandan" if id_valid else "Missing required features"
    }

    print("Final Output:", final_output)  # Print the final output to console

    return final_output
