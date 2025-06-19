import cv2
import onnxruntime
import numpy as np
from PIL import Image
from collections import deque
import time

def preprocess_image(image, size=192):
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

def get_smooth_prediction(predictions, confidences, window_size=20, confidence_threshold=0.85, min_frames=10, min_time=1.0):
    if len(predictions) < window_size:
        return predictions[-1] if predictions else None
    recent_preds = predictions[-window_size:]
    recent_confs = confidences[-window_size:]
    valid_preds = [pred for pred, conf in zip(recent_preds, recent_confs) if conf > confidence_threshold]
    if not valid_preds:
        return recent_preds[-1]
    pred_counts = {}
    for pred in valid_preds:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    max_pred = max(pred_counts.items(), key=lambda x: x[1])
    if max_pred[1] >= min_frames:
        return max_pred[0]
    else:
        return predictions[-1]

def predict_pose(video_path, onnx_path):
    session = onnxruntime.InferenceSession(onnx_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    label_map = {
        0: 'standing',
        1: 'sitting',
        2: 'lying',
        3: 'bending',
        4: 'crawling'
    }
    predictions = deque(maxlen=30)
    confidences = deque(maxlen=30)
    speed = 1.0
    current_pose = None
    last_change_time = time.time()
    min_time_between_changes = 1.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        input_data = preprocess_image(frame)
        outputs = session.run(None, {'input': input_data})
        prediction = np.argmax(outputs[0])
        confidence = outputs[0][0][prediction]
        predictions.append(prediction)
        confidences.append(confidence)
        smooth_prediction = get_smooth_prediction(
            list(predictions),
            list(confidences),
            window_size=20,
            confidence_threshold=0.85,
            min_frames=10
        )
        current_time = time.time()
        if smooth_prediction is not None and (current_pose is None or 
            (smooth_prediction != current_pose and 
             current_time - last_change_time >= min_time_between_changes)):
            current_pose = smooth_prediction
            last_change_time = current_time
        pose = label_map[current_pose] if current_pose is not None else "Unknown"
        current_confidence = outputs[0][0][current_pose] if current_pose is not None else 0.0
        cv2.putText(frame, f"Pose: {pose}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed}x", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pose Detection', frame)
        key = cv2.waitKey(int(1000/(fps*speed))) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            speed = min(speed + 0.5, 4.0)
        elif key == ord('-'):
            speed = max(speed - 0.5, 0.5)
        elif key == ord('r'):
            speed = 1.0
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test2_video.mp4"
    onnx_path = "pose_classifier.onnx"
    print("Start test...")
    print("Controls:")
    print("+ : speed up")
    print("- : slow down")
    print("r : reset speed")
    print("q : quit")
    predict_pose(video_path, onnx_path) 