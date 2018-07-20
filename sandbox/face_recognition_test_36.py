import cv2
import sys

faceCascPath = "C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(faceCascPath)

eyeCascPath = "C:\Python36\Lib\site-packages\cv2\data\haarcascade_lefteye_2splits.xml"
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

video_capture = cv2.VideoCapture(0)

while True:
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = []
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        ex,ey,ew,eh = eyes[0]
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (0,0), fx=2, fy=2)

    
    # Display the resulting frame
    cv2.imshow('Video', roi_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()