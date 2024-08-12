import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera {index} is available.")
        cap.release()
    else:
        print(f"Camera {index} is not available.")

# 测试多个摄像头
for i in range(90):
    test_camera(i)
