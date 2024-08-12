import cv2
import numpy as np
from keras.src.saving import load_model
import torchvision.transforms as transforms
import logging
import sys
from PIL import Image
from scipy.optimize import linear_sum_assignment
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class FireDetectionModel:
    _instance = None

    @staticmethod
    def get_instance(model_path):
        if FireDetectionModel._instance is None:
            FireDetectionModel(model_path)
        return FireDetectionModel._instance

    def __init__(self, model_path):
        if FireDetectionModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.model = self._load_model(model_path)
            FireDetectionModel._instance = self

    @staticmethod
    def _load_model(model_path):
        try:
            model = load_model(model_path)
            logging.info("Model loaded successfully.")
            return model
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def detect_fire(self, image):
        try:
            outputs = self.model.predict(image)
            logging.debug(f"Model outputs: {outputs}")
            fire_prob = outputs[0][1] if outputs.shape[1] == 2 else outputs[0][0]
            logging.debug(f"Fire probability: {fire_prob}")
            return fire_prob
        except Exception as e:
            logging.error(f"Error during fire detection: {e}")
            return 0.0


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    return image


def create_fire_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    return mask1 + mask2


def detect_fire_cv(frame, prev_frame):
    try:
        fire_mask = create_fire_mask(frame)
        diff = cv2.absdiff(frame, prev_frame)
        diff_mask = create_fire_mask(diff)
        combined_mask = cv2.bitwise_and(fire_mask, diff_mask)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y + h, x:x + w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, bright_mask = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
                bright_ratio = cv2.countNonZero(bright_mask) / (w * h)
                if bright_ratio > 0.1:
                    fire_regions.append((x, y, w, h, bright_ratio))
        return fire_regions
    except cv2.error as e:
        logging.error(f"Error in CV fire detection: {e}")
        return []


def calculate_iou(box1, box2):
    x1, y1, w1, h1, _ = box1
    x2, y2, w2, h2, _ = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def update_tracked_fires(tracked_fires, current_fires, iou_threshold=0.1):
    try:
        updated_fires = []
        unmatched_tracked = set(range(len(tracked_fires)))
        unmatched_current = set(range(len(current_fires)))

        if tracked_fires and current_fires:
            iou_matrix = np.zeros((len(tracked_fires), len(current_fires)), dtype=np.float32)
            for t, tracked_fire in enumerate(tracked_fires):
                for c, current_fire in enumerate(current_fires):
                    iou_matrix[t, c] = calculate_iou(tracked_fire, current_fire)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for t, c in zip(row_ind, col_ind):
                if iou_matrix[t, c] > iou_threshold:
                    updated_fires.append(current_fires[c])
                    unmatched_tracked.discard(t)
                    unmatched_current.discard(c)

        for t in unmatched_tracked:
            tracked_fires[t] = (*tracked_fires[t][:4], tracked_fires[t][4] - 1)
            if tracked_fires[t][4] > 0:
                updated_fires.append(tracked_fires[t])

        for c in unmatched_current:
            updated_fires.append(current_fires[c])

        return updated_fires
    except Exception as e:
        logging.error(f"Error updating tracked fires: {e}")
        return tracked_fires


class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        self.prediction = None

    def update(self, x, y):
        self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        self.prediction = self.kalman.predict()

    def predict(self):
        return self.kalman.predict()


def apply_nms(boxes, scores, iou_threshold=0.3):
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, b)) for b in boxes],
        scores=scores,
        score_threshold=0.5,
        nms_threshold=iou_threshold
    )
    logger.info(f"NMS indices: {indices}")
    if isinstance(indices, (np.ndarray, list)) and len(indices) > 0:
        indices = [i[0] if isinstance(i, list) else i for i in indices]
    else:
        indices = []
    return indices


def main():
    model_path = 'models/fire_smoke_detection_model.h5'
    try:
        model = FireDetectionModel.get_instance(model_path)
    except RuntimeError as e:
        logger.error(f"Failed to initialize the model. Exiting: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error opening video capture. Exiting.")
        sys.exit(1)

    _, prev_frame = cap.read()
    if prev_frame is None:
        logger.error("Error reading the initial frame. Exiting.")
        sys.exit(1)

    cv_threshold = 0.05
    ml_threshold = 0.10

    tracked_fires = []
    kalman_trackers = []
    frame_count = 0
    fire_frames = 0
    detection_history = deque(maxlen=30)
    ml_probs = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from video capture. Exiting loop.")
            break

        # Computer Vision approach
        cv_fire_regions = detect_fire_cv(frame, prev_frame)
        cv_fire_regions = [(*region[:4], 5) for region in cv_fire_regions if region[4] > cv_threshold]
        cv_fire_detected = len(cv_fire_regions) > 0

        # Machine Learning approach
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(rgb_frame)
        ml_fire_prob = model.detect_fire(image_tensor)
        ml_probs.append(ml_fire_prob)
        ml_fire_prob = np.mean(ml_probs)

        # Combine results using weighted sum
        combined_fire_prob = 0.5 * ml_fire_prob + 0.5 * cv_fire_detected
        fire_detected = combined_fire_prob > ml_threshold

        detection_history.append(fire_detected)
        smoothed_fire_detected = sum(detection_history) >= 8  # Reduced the required count for smoothing

        if smoothed_fire_detected:
            fire_frames += 1
            logger.info(
                f"Fire detected: CV Detected = {cv_fire_detected}, ML Probability = {ml_fire_prob:.2f}, Combined Probability = {combined_fire_prob:.2f}")
            boxes = [(x, y, w, h) for (x, y, w, h, _) in cv_fire_regions]
            scores = [combined_fire_prob] * len(boxes)
            indices = apply_nms(boxes, scores, iou_threshold=0.3)
            logger.info(f"Indices after NMS: {indices}")
            if indices:
                cv_fire_regions = [cv_fire_regions[i] for i in indices]
            tracked_fires = update_tracked_fires(tracked_fires, cv_fire_regions)

            # Kalman filter tracking
            for i, fire in enumerate(tracked_fires):
                x, y, w, h, _ = fire
                if i >= len(kalman_trackers):
                    kalman_trackers.append(KalmanTracker())
                kalman_trackers[i].update(x + w / 2, y + h / 2)
        else:
            tracked_fires = update_tracked_fires(tracked_fires, [])

        cv2.putText(frame, f"CV: {'Yes' if cv_fire_detected else 'No'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"ML: {ml_fire_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Combined: {combined_fire_prob:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)

        if tracked_fires:
            cv2.putText(frame, "FIRE DETECTED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for (x, y, w, h, _) in tracked_fires:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for tracker in kalman_trackers:
                prediction = tracker.predict()
                cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), -1)

        cv2.imshow('Fire Detection', frame)

        prev_frame = frame.copy()
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Exit signal received. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video capture released and all windows destroyed. Program exited.")


if __name__ == '__main__':
    main()
