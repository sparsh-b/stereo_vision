import numpy as np
import cv2

def triangulation_linear(t1, r1, t2, r2, k, src_pts, dst_pts):
    triangulation_points = []
    id_mat = np.identity(3)
    t1 = np.expand_dims(t1, 1)
    proj_mat1 = np.matmul(np.matmul(k, r1), np.hstack((id_mat, -t1)))
    proj_mat2 = np.matmul(np.matmul(k, r2), np.hstack((id_mat, -t2)))
    p11 = proj_mat1[0, :].reshape(1, 4)
    p12 = proj_mat1[1, :].reshape(1, 4)
    p13 = proj_mat1[2, :].reshape(1, 4)
    p21 = proj_mat2[0, :].reshape(1, 4)
    p22 = proj_mat2[1, :].reshape(1, 4)
    p23 = proj_mat2[2, :].reshape(1, 4)

    for i in range(len(src_pts)):
        x1, y1 = src_pts[i]
        x2, y2 = dst_pts[i]
        a_mat = []
        a_mat.append(y1*p13-p12)
        a_mat.append(p11-x1*p13)
        a_mat.append(y2*p23-p22)
        a_mat.append(p21-x2*p23)
        a_mat = np.array(a_mat).reshape(4, 4)
        _, _, V_T = np.linalg.svd(a_mat)
        triangulation_points.append(V_T[-1] / V_T[-1, -1])

    return np.array(triangulation_points)

def triangulation_points_fun(R_arr, T_arr, k, src_pts, dst_pts):
    reference_r = np.identity(3)
    reference_t = np.zeros((3, 1))
    triangulation_points_total = []
    for i in range(len(R_arr)):
        triangulation_points = triangulation_linear(T_arr[i], R_arr[i], reference_t, reference_r, k, src_pts, dst_pts)
        triangulation_points_total.append(triangulation_points)
    return triangulation_points_total
