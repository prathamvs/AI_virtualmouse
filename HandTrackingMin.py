# 1. Importations and initializations
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Object Capture for video capturing
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 2. Capturing an image input and processing it
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # 3. Working with each hand
    # checking whether the hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Working with each hand at a time
            for id, lm in enumerate(handLms.landmark):  # listed point Landmark information
                # h = height , w = width , channel
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # 4. Drawing the hand landmarks and hand connections on the hand image
                if id == 20:  # draw circle on specific spot
                    cv2.circle(image, (cx, cy), 25, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    # 5.Displaying the output
    cv2.imshow("Output", image)
    cv2.waitKey(1)
