import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions_classifier = load_model('JunH_HappyFace_3MP.h5') #input_shape= (48, 48, 1)
count = 0
file_name_number = 0
#camera.= cv2.Videocapture("") # 지정영상을 카메라로 설정
camera = cv2.VideoCapture(0) # 컴퓨터 웹캠을 카메라로 설정
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    if len(faces) > 0: #얼굴을 인식 했다면
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        roi = gray[fY: fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        pred = [0,0,0,0,0,0,0]
        count += 1
        pred = emotions_classifier.predict_step(roi)
        pred = np.argmax(emotions_classifier.predict(roi), axis=-1)
        label_map = ['anger', 'disgust', 'fear', 'happy','sad', 'surprised', 'none']
        emotionText = label_map[pred[0]]
        if emotionText == 'happy':
            cv2.putText(frame, emotionText,(fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        #if count % 5== 0: #카운트가 5의 배수일때 사진 저장/ 파일위치는 변
            #cv2.imwrite("/Users/junhyeoklee/Documents/3MP-7emotionsWebcam_Screenshot/" + str(file_name_number) + " png", frame)
            #file_name_number += 1
    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
