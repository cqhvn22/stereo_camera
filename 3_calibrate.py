"""
BUOC 3 (tuy chon nhung NEN LAM): Calibrate stereo camera bang ban co checkerboard.

Chuan bi:
    - In 1 ban co checkerboard (vi du 9x6 o vuong ben trong, tim tren Google
      "opencv checkerboard pattern pdf" de in).
    - Do kich thuoc that cua 1 o vuong (mm), sua bien SQUARE_SIZE_MM ben duoi.

Cach dung:
    python3 3_calibrate.py
    - Dua ban co ra truoc camera, thay ro CA HAI mat trai/phai.
    - Nhan phim SPACE de chup 1 cap anh mau (nen chup 15-20 cap, o nhieu
      goc do, khoang cach, vi tri khac nhau trong khung hinh).
    - Nhan 'c' de bat dau tinh calibration sau khi da chup du.
    - Nhan 'q' de thoat.

Ket qua: file stereo_calib.npz duoc luu, dung truc tiep cho code tinh depth sau nay.
"""

import cv2
import numpy as np
import os
import sys
from picamera2 import Picamera2

CHECKERBOARD = (8, 6)      # so o vuong BEN TRONG (ban co 9x7 o vuong thi so goc trong la 8x6)
SQUARE_SIZE_MM = 25.0      # kich thuoc that cua 1 o vuong (mm), do lai va sua so nay

def main():
    # Tao thu muc tam luu anh chup neu chua co
    if not os.path.exists("calib_images"):
        os.makedirs("calib_images")

    print("[+] Dang khoi tao ket noi Camera doi bang Picamera2 tren Pi 5...")
    try:
        picam0 = Picamera2(camera_num=0) # Mat camera cong so 0
        picam1 = Picamera2(camera_num=1) # Mat camera cong so 1
        
        # Thiet lap do phan giai co dinh 640x480 de toi uu toc do realtime
        config0 = picam0.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        config1 = picam1.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        
        picam0.configure(config0)
        picam1.configure(config1)
        
        picam0.start()
        picam1.start()
    except Exception as e:
        print(f"[-] Loi mo camera phan cung tren Pi 5: {e}")
        sys.exit(1)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    img_shape = None
    captured = 0

    print("\n=== HUONG DAN CHAY HIEU CHUAN ===")
    print("1. Gio ban co vao khung hinh o cac goc do va khoang cach khac nhau.")
    print("   Nhan SPACE de chup, c de tinh calibration, q de thoat.\n")

    while True:
        try:
            # chup anh tu 2 camera 
            left_rgb = picam0.capture_array()
            right_rgb = picam1.capture_array()
        except Exception as e:
            print("Khong doc duoc frame:", e)
            break

        # Chuyen tu RGB cua picamera sang BGR cho OpenCV xu ly hien thi
        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

        h, w = left.shape[:2]
        img_shape = (w, h)  # (width, height)

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(left_gray, CHECKERBOARD, None)
        found_r, corners_r = cv2.findChessboardCorners(right_gray, CHECKERBOARD, None)

        display = np.hstack([left, right]).copy()
        if found_l:
            cv2.drawChessboardCorners(display[:, :w], CHECKERBOARD, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display[:, w:], CHECKERBOARD, corners_r, found_r)

        cv2.putText(display, f"Da chup: {captured} cap", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration - trai | phai", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if found_l and found_r:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_l = cv2.cornerSubPix(left_gray, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(right_gray, corners_r, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints_left.append(corners_l)
                imgpoints_right.append(corners_r)
                
                cv2.imwrite(f"calib_images/left_{captured}.jpg", left)
                cv2.imwrite(f"calib_images/right_{captured}.jpg", right)
                
                captured += 1
                print(f"Da chup cap thu {captured}")
            else:
                print("Khong thay ro ban co o ca 2 mat, thu lai.")

        elif key == ord('c'):
            if captured < 8:
                print(f"Moi chi co {captured} cap, nen chup it nhat 8-10 cap de ket qua tot.")
                continue
            print("Dang tinh calibration, cho chut...")
            calibrate_and_save(objpoints, imgpoints_left, imgpoints_right, img_shape)
            break

        elif key == ord('q'):
            print("Thoat khong luu.")
            break

    picam0.stop()
    picam1.stop()
    cv2.destroyAllWindows()


def calibrate_and_save(objpoints, imgpoints_left, imgpoints_right, img_shape):
    # Calibrate tung camera rieng
    ret_l, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
    ret_r, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

    # Stereo calibrate de tim R, T giua 2 camera
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2, img_shape,
        criteria=criteria, flags=flags
    )
    print(f"Stereo calibration RMS error: {ret:.4f} (cang thap cang tot, <1.0 la kha ok)")

    # Tinh rectification map
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_shape, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_32FC1)

    np.savez("stereo_calib.npz",
             map1x=map1x, map1y=map1y,
             map2x=map2x, map2y=map2y,
             Q=Q, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)

    print("Da luu stereo_calib.npz thanh cong!")


if __name__ == "__main__":
    main()
