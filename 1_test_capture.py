"""
BUOC 1: Kiem tra board camera doi da hoat dong dung chua tren Raspberry Pi 5 (Picamera2).
Script nay doc luong video tu 2 camera va hien thi len man hinh.

Chay:
    python3 1_test_capture.py

Nhan 'q' de thoat.
"""

import cv2
import sys
import numpy as np
from picamera2 import Picamera2

def main():
    print("[+] Dang khoi tao ket noi Camera doi bang Picamera2 tren Pi 5...")
    try:
        picam0 = Picamera2(camera_num=0)
        picam1 = Picamera2(camera_num=1)
        
        config0 = picam0.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        config1 = picam1.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        
        picam0.configure(config0)
        picam1.configure(config1)
        
        picam0.start()
        picam1.start()
    except Exception as e:
        print(f"[-] Loi mo camera phan cung tren Pi 5: {e}")
        sys.exit(1)

    print("Dang doc camera... nhan 'q' de thoat.")

    while True:
        try:
            left_rgb = picam0.capture_array()
            right_rgb = picam1.capture_array()
        except Exception as e:
            print("Khong doc duoc frame. Kiem tra lai ket noi camera:", e)
            break

        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

        combined = np.hstack([left, right])

        cv2.imshow("Anh goc (ca 2 mat ghep)", combined)
        cv2.imshow("Camera trai", left)
        cv2.imshow("Camera phai", right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam0.stop()
    picam1.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
