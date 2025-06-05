import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # try DirectShow
    if cap.isOpened():
        print(f"Found camera at index {i}")
        cap.release()