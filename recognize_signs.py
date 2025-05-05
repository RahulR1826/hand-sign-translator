import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Tip landmarks of each finger
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = []
            landmarks = hand_landmarks.landmark
            
            # Thumb
            if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other 4 fingers
            for tip_id in finger_tips[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            total_fingers = fingers.count(1)
            
            # Display number of fingers
            cv2.putText(img, f'Fingers: {total_fingers}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Sign Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
