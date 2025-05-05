import cv2
import joblib
import numpy as np
import mediapipe as mp
import os

# Load trained model
model = joblib.load('models/hand_sign_model.pkl')

# Reply suggestion dictionary
reply_dict = {
    "Hello": "Hi",
    "Yes": "Okay",
    "No": "Why not",
    "Thank You": "You're welcome",
    "Help": "What do you need?",
    "I Love You":"Me too",
}

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        prediction = "None"
        reply = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y])
                landmark_array = np.array(landmark_list).reshape(1, -1)

                # Predict sign
                prediction = model.predict(landmark_array)[0]
                reply = reply_dict.get(prediction, "")

                # Show prediction
                cv2.putText(frame, f'Prediction: {prediction}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show suggested reply
                if reply:
                    cv2.putText(frame, f'Reply: {reply}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        else:
            cv2.putText(frame, "No Hand Detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display video feed
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

# Cleanup
cap.release()
cv2.destroyAllWindows()

