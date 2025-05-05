import cv2
import numpy as np
import mediapipe as mp
import joblib
import tkinter as tk
import threading
import os
import mss

# Load trained model
model = joblib.load('models/hand_sign_model.pkl')

# Reply suggestion dictionary
reply_dict = {
    "Hello": "Hi",
    "Yes": "Okay",
    "No": "Why not",
    "Thank You": "You're welcome",
    "Help": "What do you need?",
    "I Love You": "Me too",
}

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Extract only x and y coordinates (42 features)
def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

def recognize():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        reply = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand_landmarks)
                    if landmarks.shape[0] == 42:
                        prediction = model.predict([landmarks])[0]
                        reply = reply_dict.get(prediction, "...")
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, f'{prediction} -> {reply}', (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Sign Recognition', frame)

            key = cv2.waitKey(1) & 0xFF

            # Press 'r' to play reply sign video (Module 6)
            if key == ord('r') and reply:
                video_path = f"sign_videos/{reply}.mp4"
                if os.path.exists(video_path):
                    cap_reply = cv2.VideoCapture(video_path)
                    while cap_reply.isOpened():
                        ret_vid, frame_vid = cap_reply.read()
                        if not ret_vid:
                            break
                        cv2.imshow("Reply Sign", frame_vid)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    cap_reply.release()
                    cv2.destroyWindow("Reply Sign")
                else:
                    print(f"Video for reply '{reply}' not found!")

            # Press 'q' to quit
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def start_webcam():
    threading.Thread(target=recognize).start()


def start_screen_capture():
    def capture():
        with mss.mss() as sct:
            # Grab a full screenshot for ROI selection
            screen = sct.grab(sct.monitors[1])
            img = np.array(screen)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Let user select region
            roi = cv2.selectROI("Select Area", img, False, False)
            cv2.destroyWindow("Select Area")

            # If selection is valid
            if roi[2] > 0 and roi[3] > 0:
                monitor = {
                    "top": int(roi[1]),
                    "left": int(roi[0]),
                    "width": int(roi[2]),
                    "height": int(roi[3])
                }

                print("Press 'q' to quit screen capture.")
                while True:
                    img = np.array(sct.grab(monitor))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    cv2.imshow("Screen Capture", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()

    threading.Thread(target=capture).start()



def create_ui():
    root = tk.Tk()
    root.title("Video Source Toggle")
    root.geometry("300x200")

    webcam_btn = tk.Button(root, text="Start Webcam + Gesture", command=start_webcam, width=25)
    webcam_btn.pack(pady=5)

    screen_btn = tk.Button(root, text="Start Screen Capture", command=start_screen_capture, width=25)
    screen_btn.pack(pady=5)

    quit_btn = tk.Button(root, text="Exit", command=root.quit, width=25)
    quit_btn.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_ui()

