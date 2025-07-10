import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load("gesture_model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,10)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            if len(row) == 63:
                prediction = model.predict([row])[0]
                cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
