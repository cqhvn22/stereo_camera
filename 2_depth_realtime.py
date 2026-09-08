"""
BUOC 2: Tinh depth map (ban do do sau) real-time tu 2 camera.

Neu ban CHUA calibrate: script se chay o che do demo, dung StereoSGBM
truc tiep tren anh tho (khong rectify). Ket qua se KHONG chinh xac ve
khoang cach thuc, nhung du de thay disparity/dang bat dau.

Neu ban DA calibrate (co file stereo_calib.npz tao boi 3_calibrate.py),
script se tu dong load va rectify anh truoc khi tinh depth -> chinh xac hon nhieu.

Chay:
    python3 2_depth_realtime.py

Cac phim tat khi dang chay:
    q - thoat
    s - luu anh disparity hien tai
"""

import cv2
import numpy as np
import os

CAMERA_INDEX = 0
CALIB_FILE = "stereo_calib.npz"


def load_calibration(path):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "map1x": data["map1x"], "map1y": data["map1y"],
        "map2x": data["map2x"], "map2y": data["map2y"],
    }


def create_stereo_matcher():
    # Tham so SGBM - co the chinh de ra ket qua muot hon / chi tiet hon
    min_disp = 0
    num_disp = 16 * 8   # phai chia het cho 16
    block_size = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    return stereo, min_disp, num_disp


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Khong mo duoc camera index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    calib = load_calibration(CALIB_FILE)
    if calib:
        print(f"Da tim thay '{CALIB_FILE}' -> chay o che do CALIBRATED (chinh xac).")
    else:
        print(f"Khong tim thay '{CALIB_FILE}' -> chay o che do DEMO (chua calibrate).")
        print("Chay 3_calibrate.py truoc de co ket qua depth chinh xac.")

    stereo, min_disp, num_disp = create_stereo_matcher()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame.")
            break

        h, w = frame.shape[:2]
        mid = w // 2
        left = frame[:, :mid]
        right = frame[:, mid:]

        if calib:
            left = cv2.remap(left, calib["map1x"], calib["map1y"], cv2.INTER_LINEAR)
            right = cv2.remap(right, calib["map2x"], calib["map2y"], cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Chuan hoa de hien thi dep hon (0-255) va to mau
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        cv2.imshow("Camera trai", left)
        cv2.imshow("Depth map (disparity)", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("disparity_saved.png", disp_color)
            print("Da luu disparity_saved.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
