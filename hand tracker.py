import cv2
import mediapipe as mp
cap=cv2.VideoCapture(0)
mphands= mp.solutions.hands
hand = mphands.Hands()
draw=mp.solutions.drawing_utils
while True:
  success, image = cap.read()
  rgbimg= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  result= hand.process(rgbimg)
  if result.multi_hand_landmarks:
    for handlms in result.multi_hand_landmarks:
      for id,lm in enumerate(handlms.landmark):
        h,w,c = image.shape
        cx,cy= int(lm.x * w), int(lm.y*w)
      draw.draw_landmarks(image,handlms,mphands.HAND_CONNECTIONS)


