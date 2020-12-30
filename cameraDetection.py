
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    bottomLeft = (125, 20);
    topRight = (525,420);
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 10)
    gray = cv2.resize(gray, (280,280))
    # rect = cv2.rectangle(gray, bottomLeft, topRight, (255,0,0), 2)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Display the resulting frame
    for (x,y,w,h) in faces:
        cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
