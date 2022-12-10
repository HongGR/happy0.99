from tkinter import*
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os

def detect_face():
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotions_classifier = load_model('JunH_HappyFace_3MP.h5') #input_shape= (48, 48, 1)
    count = 49 #첫 장면을 바로 찍기 위해서 조건문의 개수인 50보다 1적은 숫자를 선언해준다.
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
            pred = emotions_classifier.predict_step(roi)
            pred = np.argmax(emotions_classifier.predict(roi), axis=-1)
            label_map = ['anger', 'disgust', 'fear', 'happy','sad', 'surprised', 'none']
            emotionText = label_map[pred[0]]
            if emotionText == 'happy':
                count += 1
                cv2.putText(frame, emotionText,(fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            if count == 50: #카운트가 150일때 사진 저장/ 파일위치는 .py있는 디렉토리
                cv2.imwrite( str(file_name_number) + ".png", frame)
                file_name_number += 1
                count=0
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

def gui():
    win = Tk() # 창 생성

    # 창 option
    win.geometry("650x500") # 창 크기 조절
    win.title("what time?") # 창 제목 짓기
    win.option_add("*Font","궁서 20") #글자 크기 미리 세팅하기

    # 버튼의 목적 : 어떤 기능의 구현, 필요한 것 : 기능을 설명해주는 문구나 기호
    btn = Button(win) #버튼 생성

    # 버튼 option
    btn.config(width =15 , height = 5) # 버튼 크기
    btn.config(text = '실행') # 버튼에 뜨는 글자
    btn.config(command = detect_face) # 버튼 기능 alert자리에 함수명 넣으면 됨.

    # 라벨 : 설명
    lab1 = Label(win)
    lab2 = Label(win)
    lab3 = Label(win)


    # 라벨 option
    lab1.config(text = '웃는 얼굴 인식하기') # 라벨에 글자넣기
    lab2.config(text = '실행을 중지하려면')
    lab3.config(text = '화면을 클릭한 후에 q키를 누르면 됩니다.')

    lab1.pack() # 라벨 배치

    btn.pack(pady= 100) # 버튼 배치

    lab2.pack()
    lab3.pack()

    win.mainloop() # 창 실행


#gui 실행
gui()
