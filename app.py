import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model


class SignLanguageTranslator:
    """
    A class for real-time hand detection and sign language translation using a CNN model.
    """

    def __init__(self):
        """
        Initialize the SignLanguageTranslator with necessary components and settings.
        """
        self.setup_streamlit()
        self.load_model()
        self.setup_mediapipe()
        self.setup_camera()
        self.index_to_value = {i: chr(65 + i) for i in range(26)}
        self.debug = True
        self.fps_deque = deque(maxlen=10)

    def setup_streamlit(self):
        """
        Configure Streamlit page settings.
        """
        st.set_page_config(
            page_title="Sign Translation Web App"
        )

    def load_cached_model(self):
        """
        Load the pre-trained CNN model for sign language classification.
        This function is cached by Streamlit to improve performance.

        Returns:
            tensorflow.keras.Model: The loaded CNN model.
        """
        return load_model('model_cnn_2.h5')

    def load_model(self):
        """
        Load the pre-trained CNN model using the cached function.
        """
        self.model_cnn = self.load_cached_model()

    def setup_mediapipe(self):
        """
        Initialize MediaPipe Hands for hand detection and tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def setup_camera(self):
        """
        Initialize the camera capture for video input.
        """
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    def detect_hands(self, image):
        """
        Detect hands in the input image and classify the hand gestures.

        Args:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            tuple: A tuple containing lists of hand bounding boxes and predictions.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        hand_boxes = []
        hand_images = []
        
        h, w, _ = image.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min = np.min([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
                x_max, y_max = np.max([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
                
                x_min, y_min = int(x_min * w), int(y_min * h)
                x_max, y_max = int(x_max * w), int(y_max * h)
                
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                hand_boxes.append((x_min, y_min, x_max, y_max))
        
        if len(hand_boxes) == 2 and self.boxes_collide(hand_boxes[0], hand_boxes[1]):
            merged_box = self.merge_boxes(hand_boxes[0], hand_boxes[1])
            hand_boxes = [merged_box]  # Replace with the merged box
        
        for box in hand_boxes:
            hand_img = image[box[1]:box[3], box[0]:box[2]]
            if hand_img.size != 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (128, 128))
                hand_img = hand_img / 255.0
                hand_images.append(hand_img.reshape(1, 128, 128, 1))
        
        predictions = []
        for hand_img in hand_images:
            pred = self.model_cnn.predict(hand_img, verbose=0)
            pred_class = np.argmax(pred)
            predictions.append((pred_class, self.index_to_value[pred_class]))
        
        return hand_boxes, predictions

    @staticmethod
    def boxes_collide(box1, box2):
        """
        Check if two bounding boxes overlap or are very close to each other.

        Args:
            box1 (tuple): First bounding box (x_min, y_min, x_max, y_max).
            box2 (tuple): Second bounding box (x_min, y_min, x_max, y_max).

        Returns:
            bool: True if boxes collide or are close, False otherwise.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        close_thresh = 50
        if (x1_min <= x2_max + close_thresh and x1_max + close_thresh >= x2_min and
            y1_min <= y2_max + close_thresh and y1_max + close_thresh >= y2_min):
            return True
        return False

    @staticmethod
    def merge_boxes(box1, box2):
        """
        Merge two bounding boxes into one.

        Args:
            box1 (tuple): First bounding box (x_min, y_min, x_max, y_max).
            box2 (tuple): Second bounding box (x_min, y_min, x_max, y_max).

        Returns:
            tuple: Merged bounding box (x_min, y_min, x_max, y_max).
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = min(x1_min, x2_min)
        y_min = min(y1_min, y2_min)
        x_max = max(x1_max, x2_max)
        y_max = max(y1_max, y2_max)
        
        return (x_min, y_min, x_max, y_max)

    def run(self):
        """
        Main method to run the sign language translation application.
        """
        st.title("Sign Translation Classification")
        
        stframe = st.empty()
        
        if self.debug:
            debug_col = st.empty()

        while True:

            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            hand_boxes, predictions = self.detect_hands(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for box, (pred_class, pred_value) in zip(hand_boxes, predictions):
                cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f'Class: {pred_value}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.fps_deque.append(1 / (time.time() - start_time))
            avg_fps = sum(self.fps_deque) / len(self.fps_deque)
            cv2.putText(frame_rgb, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            if self.debug:
                if predictions:
                    debug_col.success(f"Current detections: {[(pred_class, pred_value) for pred_class, pred_value in predictions]}")
                else:
                    debug_col.warning("No detections")
        
        self.cap.release()

if __name__ == "__main__":
    translator = SignLanguageTranslator()
    translator.run()