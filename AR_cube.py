import cv2
import numpy as np
from itertools import product
import time
import argparse


def arguments_setting():
    arguments = argparse.ArgumentParser()

    arguments.add_argument('-g', '--grid', nargs='+', type=int, required=True, default="4 4")
    arguments.add_argument('-s', '--image_size', nargs='+', type=int, required=True, default="320 240")

    args = arguments.parse_args()
    return args


def detect_checkerboard_marker(img, grid):
    found, corners = cv2.findChessboardCorners(img, grid,
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                   cv2.CALIB_CB_EXHAUSTIVE)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img, corners, grid, (-1, -1), criteria)

        return found, corners2

    return found, None


def get_world_coordinate_array(grid):
    w_coord = np.zeros((1, grid[0] * grid[1], 3), np.float32)
    w_coord[0, :, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)

    return w_coord


def get_p_camera(p_w, r, t):
    p_c = r * p_w + t
    return p_c


def get_intersection_with_2line(s1, e1, s2, e2):
    s1, e1, s2, e2 = [np.append(p,1) for p in [s1, e1, s2, e2]]

    l1 = np.cross(s1, e1)
    l2 = np.cross(s2, e2)

    kx, ky, k = np.cross(l1, l2)

    return ([kx/k, ky/k] if k != 0 else None)


def get_focal_length(markers, grid, cx, cy):
    s1 = markers[0]
    e1 = markers[grid[0] - 1]
    s2 = markers[(grid[0] - 1) * (grid[1])]
    e2 = markers[(grid[0]) * (grid[1]) - 1]

    v1 = get_intersection_with_2line(s1, e1, s2, e2)
    v2 = get_intersection_with_2line(s1, s2, e1, e2)
    # print(v1, v2)
    return np.sqrt(cx * (v1[0] + v2[0] - cx) + cy * (v1[1] + v2[1] - cy) - v1[0] * v2[0] - v1[1] * v2[1])


def get_camera_matrix(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def get_cube_edge_w(n):
    temp = []
    axis_mod = np.array([1,1,-1])
    for c in product([0, n], repeat=3):
        c = np.array(c)
        temp.append(c * axis_mod) # Z축 아래로 내려가는게 양수로 되어있는듯

    return temp


def get_cube_edge_p(cube_edges_world, R, t, c_mtx):
    cube_edges_pixel = []
    cube_edges_pixel_int = []

    f = c_mtx[0][0]
    cx = c_mtx[0][2]
    cy = c_mtx[1][2]

    for i in range(len(cube_edges_world)):
        x_c = np.matmul(R, cube_edges_world[i]) + np.squeeze(t)  # 카메라 좌표
        x_p = [x_c[0] / x_c[2] * f + cx, x_c[1] / x_c[2] * f + cy]
        cube_edges_pixel.append(x_p)

    for v in cube_edges_pixel:
        cube_edges_pixel_int.append(list(map(int, v)))

    return cube_edges_pixel_int


def main():
    args = arguments_setting()
    # i_size = (320, 240)
    # grid = (4, 4)

    # principal point 는 이미지 중심점 사용
    cx, cy = args.image_size[1] / 2 - 1, args.image_size[1] / 2 - 1

    vid = cv2.VideoCapture(0)
    # frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 1
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        img_color = cv2.resize(frame, dsize=args.image_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # cube 만들기 시작

        # 월드 좌표 배열 생성
        world_coord_arr = get_world_coordinate_array(args.grid)

        # checkerboard marker 찾기
        found, markers = detect_checkerboard_marker(img, grid=args.grid)

        # checkerboard marker 를 찾지못하는 경우 예외처리
        if not found:
            cv2.imshow("test", img_color)
            if cv2.waitKey(1) & 0xFF == 27:
                vid.release()
                cv2.destroyAllWindows()
                break
            continue

        # focal length, camera matrix 구하기
        # 굳이 매 프레임마다 구할 필요가 있을까 싶어서 30프레임마다 구하도록 조건줌
        if count % 30 == 1:
            f = get_focal_length(markers, args.grid, cx, cy)

            c_mtx = get_camera_matrix(f, cx, cy)

        dist_coeffs = np.zeros((4, 1))  # 카메라 왜곡 무시

        # estimate camera pose
        ret2, r_vec, t_vec = cv2.solvePnP(world_coord_arr, markers, c_mtx, dist_coeffs)

        # solvePnP 로 값이 안나오는 경우 예외처리
        if ret2 and (np.isnan(r_vec).sum() == 0 and np.isnan(t_vec).sum() == 0):
            R = cv2.Rodrigues(r_vec)[0]  # 3x3 rotation matrix
            t = t_vec  # translation matrix
        else:
            cv2.imshow("test", img_color)
            if cv2.waitKey(1) & 0xFF == 27:
                vid.release()
                cv2.destroyAllWindows()
                break
            continue

        # cube 의 월드 좌표, 픽셀 좌표 정의
        cube_edges_world = get_cube_edge_w(args.grid[0] - 1)

        cube_edges_pixel_int = get_cube_edge_p(cube_edges_world, R, t, c_mtx)

        # cube 그리기
        for i in range(0, len(cube_edges_pixel_int), 2):
            cv2.line(img_color, cube_edges_pixel_int[i], cube_edges_pixel_int[i + 1], color=(255, 255, 0))

        cv2.line(img_color, cube_edges_pixel_int[1], cube_edges_pixel_int[3], color=(255, 255, 0), thickness=2)
        cv2.line(img_color, cube_edges_pixel_int[1], cube_edges_pixel_int[5], color=(255, 255, 0), thickness=2)
        cv2.line(img_color, cube_edges_pixel_int[-1], cube_edges_pixel_int[3], color=(255, 255, 0), thickness=2)
        cv2.line(img_color, cube_edges_pixel_int[-1], cube_edges_pixel_int[5], color=(255, 255, 0), thickness=2)

        cv2.drawChessboardCorners(img_color, args.grid, markers, found)

        cv2.imshow("test", img_color)

        if cv2.waitKey(1) & 0xFF == 27:
            vid.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

    cv2.destroyAllWindows()