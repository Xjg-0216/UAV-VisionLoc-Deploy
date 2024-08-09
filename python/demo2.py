import cv2

# /dev/video81 通常对应的索引是 81
cap = cv2.VideoCapture(81)

if not cap.isOpened():
    print("无法打开摄像头 81")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
