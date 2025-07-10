import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils
gesture_label = "no"  
filename = "gesture_data.csv"
cap = cv2.VideoCapture(0)

# Open CSV file for appending data
with open(filename, mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to RGB (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Draw hand landmarks and collect data
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 21 landmarks → each has x, y, z
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                # Append label at the end
                row.append(gesture_label)

                # Write to CSV
                writer.writerow(row)

        # Show the image
        cv2.imshow("Collecting Hand Gesture Data", frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
