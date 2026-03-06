import cv2
import os
import pyautogui
import time
import numpy as np

# use the new MediaPipe Tasks API for Python 0.10+
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    hand_landmarker,
    drawing_utils,
)
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode


class GestureKeyboard:
    def __init__(self):
        # download or locate the hand landmarker model bundle
        self.model_path = os.path.join(os.getcwd(), "hand_landmarker.task")
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker.task model...")
            try:
                import urllib.request

                url = (
                    "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
                )
                urllib.request.urlretrieve(url, self.model_path)
                print("Model downloaded to", self.model_path)
            except Exception as e:
                raise RuntimeError("Failed to download MediaPipe model: " + str(e))

        # create hand landmarker using the tasks API
        base_opts = base_options.BaseOptions(model_asset_path=self.model_path)
        opts = HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(opts)

        # drawing helpers
        self.draw_utils = drawing_utils
        self.hand_connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Gesture detection parameters
        self.touch_threshold = 0.05  # Normalized distance threshold for touch detection
        self.cooldown = 0.5  # Seconds between repeated gestures
        self.last_gesture_time = {}  # Track last time each gesture was triggered

        # Finger landmark indices (MediaPipe hand landmarks)
        self.finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

        self.left_gestures = {
            (4, 8): 'a',   # index-middle
            (4, 12): 'b',    # thumb-index
            (4, 16): 'c',   # thumb-middle
            (4, 20): 'd',   # thumb-ring
            (4, 6): 'e',   # thumb-pinky
            (4, 10): 'f',   # index-ring
            (4, 14): 'g',   # index-pinky
            (4, 18): 'h',  # middle-ring
            (8, 2): 'i',  # middle-pinky
            (12, 2): 'j',  # ring-pinky
            (16, 2): 'k',  # thumb-index-middle
            (17, 2): 'l',  # thumb-index-ring
        }

        self.right_gestures = {
            (4, 8): 'm',   # index-middle
            (4, 12): 'n',    # thumb-index
            (4, 20): 'p',   # thumb-ring
            (4, 6): 'r',   # index-ring
            (4, 10): 't',   # index-pinky
            (4, 16): 'z',  # middle-ring
            (12, 20): ' ',  # middle-pinky
            (4, 17): 'backspace',  # ring-pinky
        }

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def detect_touches(self, landmarks):
        """Detect which fingertip pairs are touching based on distance threshold."""
        touches = []
        for i in range(len(self.finger_tips)):
            for j in range(i+1, len(self.finger_tips)):
                tip1 = self.finger_tips[i]
                tip2 = self.finger_tips[j]
                dist = self.calculate_distance(landmarks[tip1], landmarks[tip2])
                if dist < self.touch_threshold:
                    touches.append((tip1, tip2))
        return touches

    

    def map_gesture(self, touches, hand_label):
        """Map detected touches to keyboard input based on hand side."""
        # Check for special single-touch gestures
        if len(touches) == 1:
            touch = touches[0]
            if hand_label == 'Left' and touch == (4, 20):  # thumb-pinky
                return 'space'
            elif hand_label == 'Right' and touch == (4, 20):  # thumb-pinky
                return 'backspace'

        # Check for letter gestures
        mapping = self.left_gestures if hand_label == 'Left' else self.right_gestures
        for touch in touches:
            if touch in mapping:
                return mapping[touch]

        return None

    def run(self):
        """Main loop for gesture detection and keyboard input."""
        print("Starting Finger Gesture Keyboard...")
        print("Press 'q' in the video window to quit.")
        print("Make gestures with your hands to type.")

        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from webcam.")
                break

            # prepare frame for the hand landmarker
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.Image(
                image_format=mp_image.ImageFormat.SRGB, data=img_rgb
            )

            current_time = time.time()
            detected_gesture = None

            # run detection
            result = self.hand_landmarker.detect(mp_img)

            if result.hand_landmarks:
                # iterate hands
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    hand_label = result.handedness[hand_idx][0].category_name

                    # draw landmarks on the original BGR image
                    self.draw_utils.draw_landmarks(
                         img, landmarks, self.hand_connections
                    )

                    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
                    touches = self.detect_touches(coords)
                    gesture = self.map_gesture(touches, hand_label)

                    if gesture:
                        detected_gesture = gesture
                        if (
                            gesture not in self.last_gesture_time
                            or current_time - self.last_gesture_time[gesture]
                            > self.cooldown
                        ):
                            pyautogui.press(gesture)
                            self.last_gesture_time[gesture] = current_time
                            print(f"Pressed: {gesture}")

                # inter-hand gesture check
                if len(result.hand_landmarks) == 2:
                    left_idx = 0 if result.handedness[0][0].category_name == 'Left' else 1
                    right_idx = 1 - left_idx
                    left_coords = [
                        (lm.x, lm.y, lm.z)
                        for lm in result.hand_landmarks[left_idx]
                    ]
                    right_coords = [
                        (lm.x, lm.y, lm.z)
                        for lm in result.hand_landmarks[right_idx]
                    ]
                    dist = self.calculate_distance(left_coords[8], right_coords[8])
                    if dist < 0.15:
                        gesture = 'enter'
                        if (
                            gesture not in self.last_gesture_time
                            or current_time - self.last_gesture_time[gesture]
                            > self.cooldown
                        ):
                            pyautogui.press('enter')
                            self.last_gesture_time[gesture] = current_time
                            detected_gesture = gesture
                            print(f"Pressed: {gesture}")

            if detected_gesture:
                cv2.putText(img, f"Gesture: {detected_gesture.upper()}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(img, "Press 'q' to quit", (10, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the image
            cv2.imshow("Finger Gesture Keyboard", img)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        try:
            self.hand_landmarker.close()
        except Exception:
            pass
        print("Gesture Keyboard stopped.")

if __name__ == "__main__":
    keyboard = GestureKeyboard()
    keyboard.run()






