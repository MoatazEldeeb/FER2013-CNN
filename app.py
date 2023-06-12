import cv2
import sys
from tensorflow import keras
import numpy as np
from datetime import datetime

model = keras.models.load_model("DeepLearning-Project-VGG16.h5")


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE,
    )

    # Draw a rectangle around the faces

    for x, y, w, h in faces:
        try:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = frame[y : y + h, x : x + w]
            crop_img = cv2.resize(crop_img, (197, 197))
            crop_img = np.array(crop_img).reshape((1, 197, 197, 3))
            pred = model.predict(crop_img)
            pred = EMOTIONS_LIST[np.argmax(pred)]
            cv2.putText(frame, pred, (x, y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(str(e))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
        cv2.imwrite("./preview/frame-" + str(datetime.now()) + ".jpg", crop_img)

    cv2.imshow("Video", frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
