import cv2

FACE = '/Users/nicknorden/Downloads/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
EYES ='/Users/nicknorden/Downloads/opencv3-3.1.0-py36_0/share/OpenCV/haarcascades/haarcascade_eye.xml'

webcam = cv2.VideoCapture(0)
cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
cv2.resizeWindow('preview',500,500)
face_cascade = cv2.CascadeClassifier(FACE)
eye_cascade = cv2.CascadeClassifier(EYES)


if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    rval = False

while rval:
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(grey,(x,y),(x+w,y+h),(0,0,0),2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
        for (ex,ey,ew,eh) in eyes[:2]:
            cv2.rectangle(roi_grey,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

    cv2.putText(grey, "Press ESC to close.", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0))
    cv2.imshow("preview", grey)

    # get next frame
    rval, frame = webcam.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break


cap.release()
out.release()
cv2.destroyAllWindows()
