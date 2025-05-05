import cv2
import mediapipe as mp
import csv
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(0)

# Create/Open CSV file to save data
csv_file = open('hand_signs_data.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Ask user for the label (sign name)
label = input("Enter the label for this hand sign: ")

print("[INFO] Starting data collection in 5 seconds...")
time.sleep(5)
print("[INFO] Collecting data now...")

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append(lm.x)
                lm_list.append(lm.y)

            # Add label at end of landmarks
            lm_list.append(label)

            # Save to CSV
            csv_writer.writerow(lm_list)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
