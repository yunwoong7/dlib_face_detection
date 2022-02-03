from imutils import face_utils
import dlib
import cv2
import face_recognition

detector = dlib.get_frontal_face_detector()
predictor = face_recognition.api.pose_predictor_68_point

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for (i, rect) in enumerate(rects):
        # 얼굴 영역의 얼굴 랜드마크를 결정한 다음
        # 얼굴 랜드마크(x, y) 좌표를 NumPy Array로 변환합니다.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # dlib의 사각형을 OpenCV bounding box로 변환(x, y, w, h)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 얼굴 랜드마크에 포인트를 그립니다.
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()