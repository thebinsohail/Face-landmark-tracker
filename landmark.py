import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX 
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(28, 36):
            nose_x = landmarks.part(n).x
            nose_y = landmarks.part(n).y
            cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)
            if landmarks==None:
             print('No Face Detected')


        cv2.putText(frame,'nose',(nose_x,nose_y),font,1,(0,255,0))

            
        for n in range(48, 68):
            mouth_x = landmarks.part(n).x
            mouth_y = landmarks.part(n).y
            cv2.circle(frame, (mouth_x, mouth_y), 4, (0, 255, 255), -1)

        cv2.putText(frame,'mouth',(mouth_x,mouth_y+50),font,1,(0,255,255))
      

        for n in range(37, 48):
            eye_x = landmarks.part(n).x
            eye_y = landmarks.part(n).y
            cv2.circle(frame, (eye_x, eye_y), 4, (255, 255, 255), -1)

        cv2.putText(frame,'eyes',(eye_x,eye_y-50),font,1,(255,255,255))
   
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break