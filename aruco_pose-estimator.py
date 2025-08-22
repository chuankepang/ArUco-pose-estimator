"""
ArUco 6D 位姿估计 (支持双目相机)
原始逻辑保持不变，只做结构化整理
"""

import cv2
import numpy as np
import glob
import os
import csv
from tqdm import tqdm


# -------------------------------
# 全局配置
# -------------------------------
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# 相机参数
height, width = 2000, 1440

fx_l, fy_l, tao_l, cx_l, cy_l = 4508.847, 4508.009, 0, 728.389, 993.697
k1_l, k2_l, p1_l, p2_l, k3_l = -0.221, 0.0823, 0, 0, 0

fx_r, fy_r, tao_r, cx_r, cy_r = 4503.550, 4503.205, 0, 715.019, 986.327
k1_r, k2_r, p1_r, p2_r, k3_r = -0.225, 0.283, 0, 0, 0

R_r2l = np.array([0.974, 0.005, -0.225,
                  -0.005, 0.999, 0.000,
                   0.225, 0.001, 0.974]).reshape(3, 3).T
T_r2l = np.array([-240.341, 0.272, 18.247]).T


# -------------------------------
# 相机类
# -------------------------------
class StereoCamera:
    def __init__(self):
        self.height, self.width = height, width
        self.K_left = np.array([[fx_l, tao_l, cx_l],
                                [0,     fy_l, cy_l],
                                [0,        0,    1]])
        self.K_right = np.array([[fx_r, tao_r, cx_r],
                                 [0,     fy_r, cy_r],
                                 [0,        0,    1]])

        self.dist_left = np.array([[k1_l, k2_l, p1_l, p2_l, k3_l]])
        self.dist_right = np.array([[k1_r, k2_r, p1_r, p2_r, k3_r]])

        self.R = R_r2l
        self.T = T_r2l


# -------------------------------
# 功能函数
# -------------------------------
def aruco_display(corners, ids, rejected, image):
    """绘制 ArUco 检测结果"""
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            pts = [tuple(map(int, topLeft)),
                   tuple(map(int, topRight)),
                   tuple(map(int, bottomRight)),
                   tuple(map(int, bottomLeft))]

            cv2.polylines(image, [np.array(pts)], True, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"[Inference] ArUco marker ID: {markerID}")

    return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """单目 ArUco 位姿估计"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    if len(corners) > 0:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.02, matrix_coefficients, distortion_coefficients)

            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame


# -------------------------------
# 主执行逻辑
# -------------------------------
def offline_aruco_pipeline():
    """主逻辑：双目标定 + ArUco 检测 + 位姿估计"""

    # 输入输出路径
    left_folder = 'input/aruco/20250630/L'
    right_folder = 'input/aruco/20250630/R'
    output_root = 'output/aruco/20250630'

    # ArUco配置
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    marker_size_m = 0.01  # 标志物边长（米）

    cam = StereoCamera()
    camera_matrix_left, dist_coeffs_left = cam.K_left, cam.dist_left
    camera_matrix_right, dist_coeffs_right = cam.K_right, cam.dist_right
    R, T = cam.R, cam.T.reshape(3, 1)

    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 输出文件夹
    os.makedirs(output_root, exist_ok=True)
    vis_l_dir = os.path.join(output_root, 'vis/l')
    vis_r_dir = os.path.join(output_root, 'vis/r')
    vis_cat_dir = os.path.join(output_root, 'vis/cat')
    for d in [vis_l_dir, vis_r_dir, vis_cat_dir]:
        os.makedirs(d, exist_ok=True)

    # 读取图像
    left_images = sorted(glob.glob(os.path.join(left_folder, '*.bmp')))
    right_images = sorted(glob.glob(os.path.join(right_folder, '*.bmp')))
    assert len(left_images) == len(right_images), '图像数量不一致'

    # 投影矩阵
    P1 = camera_matrix_left.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = camera_matrix_right.dot(np.hstack((R, T)))

    marker_tracks = {}

    # 遍历图像
    for idx, (lp, rp) in enumerate(tqdm(zip(left_images, right_images),
                                        total=len(left_images), desc='Frames')):
        img_l, img_r = cv2.imread(lp), cv2.imread(rp)
        gray_l, gray_r = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ul = cv2.undistort(gray_l, camera_matrix_left, dist_coeffs_left)
        ur = cv2.undistort(gray_r, camera_matrix_right, dist_coeffs_right)
        vis_l, vis_r = cv2.cvtColor(ul, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ur, cv2.COLOR_GRAY2BGR)

        corners_l, ids_l, _ = detector.detectMarkers(ul)
        corners_r, ids_r, _ = detector.detectMarkers(ur)

        if ids_l is None or ids_r is None:
            continue
        ids_l, ids_r = ids_l.flatten(), ids_r.flatten()

        common = set(ids_l).intersection(ids_r)
        centers = {}

        for marker_id in common:
            idx_l, idx_r = list(ids_l).index(marker_id), list(ids_r).index(marker_id)
            pts_l, pts_r = corners_l[idx_l].reshape(4, 2), corners_r[idx_r].reshape(4, 2)

            obj_pts = np.array([[0, 0, 0], [marker_size_m, 0, 0],
                                [marker_size_m, marker_size_m, 0], [0, marker_size_m, 0]], dtype=np.float32)

            ret, rvec, tvec = cv2.solvePnP(obj_pts, pts_l.astype(np.float32), camera_matrix_left, dist_coeffs_left)
            if not ret:
                continue

            # 三角化
            c_l, c_r = np.mean(pts_l, axis=0).reshape(2, 1), np.mean(pts_r, axis=0).reshape(2, 1)
            pts4d = cv2.triangulatePoints(P1, P2, c_l, c_r)
            pts4d /= pts4d[3]
            X, Y, Z = pts4d[:3].flatten()

            # rvec -> 欧拉角
            Rm, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(Rm[0, 0] ** 2 + Rm[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                roll, pitch, yaw = np.arctan2(Rm[2, 1], Rm[2, 2]), np.arctan2(-Rm[2, 0], sy), np.arctan2(Rm[1, 0], Rm[0, 0])
            else:
                roll, pitch, yaw = np.arctan2(-Rm[1, 2], Rm[1, 1]), np.arctan2(-Rm[2, 0], sy), 0

            marker_tracks.setdefault(marker_id, []).append((idx, X, Y, Z, roll, pitch, yaw))

            centers[marker_id] = (int(c_l.item(0)), int(c_l.item(1)), int(c_r.item(0)), int(c_r.item(1)))

            # 可视化
            cv2.aruco.drawDetectedMarkers(vis_l, corners_l, ids_l)
            cv2.aruco.drawDetectedMarkers(vis_r, corners_r, ids_r)
            cv2.drawFrameAxes(vis_l, camera_matrix_left, dist_coeffs_left, rvec, tvec, 0.02)
            cv2.putText(vis_l, f"ID:{marker_id}", tuple(pts_l[0].astype(int) - [0, 10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 拼接图像
        h, w = vis_l.shape[:2]
        stereo = np.hstack((vis_l, vis_r))
        for marker_id, (xl, yl, xr, yr) in centers.items():
            cv2.line(stereo, (xl, yl), (xr + w, yr), (0, 255, 0), 1)

        cv2.imwrite(os.path.join(vis_l_dir, f'vis_l_{idx:04d}.jpg'), vis_l)
        cv2.imwrite(os.path.join(vis_cat_dir, f'stereo_{idx:04d}.jpg'), stereo)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # 保存 CSV
    for mid, track in marker_tracks.items():
        d = os.path.join(output_root, f'marker_{mid}')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'{mid}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'X', 'Y', 'Z', 'roll', 'pitch', 'yaw'])
            w.writerows(track)

    print(f'完成，结果保存在 {output_root}')


# -------------------------------
# 主入口
# -------------------------------
if __name__ == "__main__":
    offline_aruco_pipeline()