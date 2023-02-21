from facial_emotion_recognition import EmotionRecognition
import cv2
e = EmotionRecognition(device='CPU')
cam = cv2.VideoCapture(0)
while True:
    success, frame = cam.read()
    frame = e.recognise_emotion(frame, return_type = 'BGR')
    #print("The Given Emotion is: ",frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()