import cv2

cap = cv2.VideoCapture(0) # 카메라 설정
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #정면 얼굴 casacdaClassifier

while ( cap.isOpened() ):
    ret, frame = cap.read() # 카메라 읽기

    if ret: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame BGR을 GRAY로 변경
        faces = face_cascade.detectMultiScale(gray, 1.1, 6) # 얼굴 분류

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2) # 얼굴 인식을 표현하기 위해 얼굴에 사각형 표시
        
        cv2.imshow('Video',frame) # 화면 띄우기

        # Press q on keyboard or exit()
        if cv2.waitKey(25) & 0xFF == ord('q'): # 종료
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()