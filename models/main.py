import cv2
import os

# Setup paths using os.path for proper path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

# Create models directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Update model paths
faceProto = os.path.join(models_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(models_dir, "age_deploy.prototxt")
ageModel = os.path.join(models_dir, "age_net.caffemodel")

# Face detection function remains the same
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxes = []
    
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return frame, bboxes

# Load models with better error handling
try:
    # Check if model files exist
    required_files = [
        (faceProto, "Face Proto"),
        (faceModel, "Face Model"),
        (ageProto, "Age Proto"),
        (ageModel, "Age Model")
    ]
    
    missing_files = [f[1] for f in required_files if not os.path.exists(f[0])]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    print("Models loaded successfully!")

except Exception as e:
    print(f"Error: {str(e)}")
    print(f"\nPlease ensure model files are in: {models_dir}")
    exit(1)

# Define age ranges and model parameters
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (104, 117, 123)

# Initialize video capture - try different indices if 0 doesn't work
try:
    video = cv2.VideoCapture(0)  # Try without CAP_DSHOW first
    if not video.isOpened():
        video = cv2.VideoCapture(1)  # Try second camera
    if not video.isOpened():
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try with DirectShow
    if not video.isOpened():
        raise RuntimeError("Could not open any camera")
    
    print("Camera opened successfully!")
except Exception as e:
    print(f"Error opening camera: {e}")
    exit(1)

try:
    if not video.isOpened():
        raise RuntimeError("Could not open video capture")
    
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame, bboxes = faceBox(faceNet, frame)
        
        for bbox in bboxes:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]
            label = f"Age: {age}"
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Age Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video.release()
    cv2.destroyAllWindows()

def test_camera():
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return False
    
    print("Camera test successful!")
    cap.release()
    return True

if __name__ == "__main__":
    test_camera()
