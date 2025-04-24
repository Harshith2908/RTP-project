import cv2
import os
import sys
import numpy as np
from typing import Tuple, List
from datetime import datetime

# === Constants ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
CONFIDENCE_THRESHOLD = 0.6


def log(msg: str):
    """Timestamped log."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_models() -> Tuple[cv2.dnn.Net, cv2.dnn.Net]:
    """Load face detection and age prediction models."""
    paths = {
        "face_prototxt": os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt"),
        "face_model": os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb"),
        "age_prototxt": os.path.join(MODELS_DIR, "age_deploy.prototxt"),
        "age_model": os.path.join(MODELS_DIR, "age_net.caffemodel")
    }

    # Check for missing files
    for key, path in paths.items():
        if not os.path.exists(path):
            log(f"❌ Missing model file: {path}")
            sys.exit(1)

    # Load models
    face_net = cv2.dnn.readNet(paths["face_model"], paths["face_prototxt"])
    age_net = cv2.dnn.readNet(paths["age_model"], paths["age_prototxt"])

    log("✅ Models loaded successfully")
    return face_net, age_net


def detect_faces(net: cv2.dnn.Net, frame: np.ndarray) -> List[List[int]]:
    """Detect faces and return bounding boxes."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def predict_age(net: cv2.dnn.Net, face_img: np.ndarray) -> Tuple[str, float]:
    """Predict age and return label and confidence."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()
    return AGE_LIST[idx], preds[0][idx]


def draw_label(frame: np.ndarray, box: List[int], label: str):
    """Draw label and box on frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_video_input(face_net, age_net):
    """Run real-time webcam age detection."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log("❌ Could not access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log("📷 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log("❌ Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        bboxes = detect_faces(face_net, frame)

        for box in bboxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            age, conf = predict_age(age_net, face)
            draw_label(frame, box, f"{age} ({conf*100:.1f}%)")

        cv2.imshow("🧠 Age Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log("✅ Webcam closed. Exiting...")


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        log(f"📁 Created models folder at {MODELS_DIR}")

    face_net, age_net = load_models()
    process_video_input(face_net, age_net)


if __name__ == "__main__":
    main()
