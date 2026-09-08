"""
BUOC 1: Kiem tra board camera doi da hoat dong dung chua.
Board Arducam Synchronized Stereo HAT xuat ra 1 luong video DUY NHAT,
trong do anh trai va anh phai duoc ghep canh nhau (side-by-side).
Script nay doc luong do va tach lam 2 de xem thu.

Chay:
    python3 1_test_capture.py

Nhan 'q' de thoat.
"""

import cv2
import sys

# Neu Raspberry Pi dung camera qua /dev/video0, doi so 0 thanh dung index cua ban.
# Neu dung libcamera/picamera2 thay vi OpenCV VideoCapture, bao minh de doi code.
CAMERA_INDEX = 0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Khong mo duoc camera index {CAMERA_INDEX}.")
        print("Kiem tra: da bat camera interface chua (raspi-config), "
              "cap FPC da cam dung chieu chua, chay 'ls /dev/video*' de xem cac index co san.")
        sys.exit(1)

    # Thu dat do phan giai pho bien cho stereo HAT (thuong la 2x chieu rong 1 camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Dang doc camera... nhan 'q' de thoat.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame. Kiem tra lai ket noi camera.")
            break

        h, w = frame.shape[:2]
        mid = w // 2

        left = frame[:, :mid]
        right = frame[:, mid:]

        cv2.imshow("Anh goc (ca 2 mat ghep)", frame)
        cv2.imshow("Camera trai", left)
        cv2.imshow("Camera phai", right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
