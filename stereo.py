import copy
from tqdm.auto import tqdm
from itertools import combinations
from math import factorial, sqrt
import cv2
import numpy as np
from cv2.xfeatures2d import SIFT_create

def get_cam_mat(file_path):
    cam_mat0 = np.array([[0,0,0],[0,0,0],[0,0,1.]])
    lines = open(file_path, 'r').readlines()
    cam0 = lines[0][6:-1].split(' ')
    cam_mat0[0,0] = float(cam0[0])
    cam_mat0[0,2] = float(cam0[2][:-1])
    cam_mat0[1,1] = cam_mat0[0,0]
    cam_mat0[1,2] = float(cam0[5][:-1])
    return cam_mat0

def normalize(src_pts, dst_pts):
    norm_src_pts = copy.copy(src_pts)
    norm_dst_pts = copy.copy(dst_pts)
    
    src_mean = np.mean(norm_src_pts, axis=0)
    dst_mean = np.mean(norm_dst_pts, axis=0)
    norm_src_pts -= src_mean
    norm_dst_pts -= dst_mean
    src_scale = sqrt(2)/sqrt(np.mean(np.sum(np.square(norm_src_pts), axis=1)))
    dst_scale = sqrt(2)/sqrt(np.mean(np.sum(np.square(norm_dst_pts), axis=1)))
    Ta = np.matmul([[src_scale,0,0], [0,src_scale,0], [0,0,1]], [[1,0,-src_mean[0]], [0,1,-src_mean[1]], [0,0,1]])
    Tb = np.matmul([[dst_scale,0,0], [0,dst_scale,0], [0,0,1]], [[1,0,-dst_mean[0]], [0,1,-dst_mean[1]], [0,0,1]])
    for i in range(norm_src_pts.shape[0]):
        norm_src_pts[i] = np.matmul(Ta, np.concatenate((src_pts[i], np.array([1]))))[:-1]
        norm_dst_pts[i] = np.matmul(Tb, np.concatenate((dst_pts[i], np.array([1]))))[:-1]
    return norm_src_pts, norm_dst_pts, Ta, Tb

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    print(img1.shape)
    r,c,ch = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 2)
        img1 = cv2.circle(img1,tuple(pt1.astype(int)),5,color,4)
        img2 = cv2.circle(img2,tuple(pt2.astype(int)),5,color,4)
    return img1,img2

def draw_epipolar_lines(img1,img2,src_pts,dst_pts,F):
    lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    
    img5,img6 = drawlines(img1,img2,lines1,src_pts,dst_pts)

    lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, dst_pts, src_pts)

    print(lines1.shape, lines2.shape)
    # cv2.imshow('img5', img5)
    # cv2.imshow('img3', img3)
    cv2.imwrite('data/{}/unnormalized_epipolar_img0.jpg'.format(dataset), img5)
    cv2.imwrite('data/{}/unnormalized_epipolar_img1.jpg'.format(dataset), img3)
    cv2.waitKey(0)

def create_F_matrix(selected_pts): #x' & x2 are equivalent
    A = []
    for i in range(8):
        x1,y1 = selected_pts[0][i]
        x2,y2 = selected_pts[1][i]
        A_row = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        A.append(A_row)
    A = np.array(A)
    U, S, V_T = np.linalg.svd(A)
    F_ = V_T[-1]
    F = F_.reshape((3,3))
    #print(V_T, F_, F)
    return F

def ransac_for_F(selected_pts, epsilon, Ta, Tb):
    list_size = selected_pts[0].shape[0]
    list_points = [i for i in range(list_size)]
    all_combinations = combinations(list_points, 8)
    num_combs = factorial(list_size)/(factorial(list_size-8)*factorial(8))
    max_inliers = 0
    num_max_inliers = 0
    prev_max_inliers = 0
    #with tqdm(total = list_size) as pbar:
    min_sum_dist = 1e5
    for i in tqdm(all_combinations):
        F_norm = create_F_matrix([selected_pts[0][list(i)], selected_pts[1][list(i)]])
        # inliers = 0
        # unselected_el = [j for j in [k for k in range(list_size)] if j not in i]
        sum_dist = 0
        for j in range(list_size):#unselected_el
            x2 = np.concatenate((selected_pts[1][j], np.array([1])))
            x1 = np.concatenate((selected_pts[0][j], np.array([1])))
            x2fx1 = abs(np.matmul(np.matmul(x2, F_norm), x1))
            sum_dist += x2fx1
            #if x2fx1 < epsilon:
            #    inliers += 1
        #print(inliers)
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            F_norm_ransac = F_norm
            # F_ransac = F
        '''
        if inliers > max_inliers:
            prev_max_inliers = max_inliers
            max_inliers = inliers
            F_ransac = [F]

        if inliers == max_inliers:
            num_max_inliers += 1
            #print(inliers)
        #pbar.update(1)
        '''
        
    print('\n', min_sum_dist)#, max_inliers, num_max_inliers)#F_ransac)
    U, S, V_T  = np.linalg.svd(F_norm_ransac)
    S = np.diag(S)
    S[-1,-1] = 0#Forcing rank of S to be 2, to account for noise
    F_norm_ransac = np.matmul(np.matmul(U, S), V_T)
    F_ransac = np.matmul(np.matmul(Tb.T, F_norm_ransac), Ta)

    # F_ransac = np.divide(F_ransac, F_ransac[-1, -1])
    # F_norm_ransac = np.divide(F_norm_ransac, F_norm_ransac[-1, -1])
    # print(F_ransac, S_ransac)
    return F_ransac

def estimate_essential_mat(F, K):
    E = np.matmul(np.matmul(K.T, F), K)
    U, S, V_T = np.linalg.svd(E)
    S = np.diag([1,1,0]) #Forcing rank of S to be 2, to account for noise
    E = np.matmul(np.matmul(U, S), V_T)
    return E

def decompose_E(E):
    U, _, V_T = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    c1 = U[:, -1]
    r1 = np.matmul(np.matmul(U, W), V_T)
    c2 = -U[:, -1]
    r2 = np.matmul(np.matmul(U, W), V_T)
    c3 = U[:, -1]
    r3 = np.matmul(np.matmul(U, W.T), V_T)
    c4 = -U[:, -1]
    r4 = np.matmul(np.matmul(U, W.T), V_T)
    rs = [r1, r2, r3, r4]
    cs = [c1, c2, c3, c4]
    for i in range(len(rs)):
        if np.linalg.det(rs[i]) < 0:
            rs[i] = -rs[i]
            cs[i] = -cs[i]
    return rs, cs

if __name__ == '__main__':
    dataset = 'curule' #octagon #pendulum
    img1 = cv2.imread('data/{}/im0.png'.format(dataset))
    img2 = cv2.imread('data/{}/im1.png'.format(dataset))
    K = get_cam_mat('data/{}/calib.txt'.format(dataset))

    sift = SIFT_create()
    key_points1, descriptors1 = sift.detectAndCompute(img1,None) # keypoints coordinates & keypoint descriptors
    key_points2, descriptors2 = sift.detectAndCompute(img2,None)
    flann_idx = 0
    index_params = dict(algorithm = flann_idx, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.17*n.distance:#c0.2 o0.10 p0.15
            good.append(m)
    assert len(good)>=8

    src_pts = np.squeeze(np.float32([key_points1[m.queryIdx].pt for m in good ]).reshape(-1,1,2))
    dst_pts = np.squeeze(np.float32([key_points2[m.trainIdx].pt for m in good ]).reshape(-1,1,2))
    # np.save(open('src_pts.npy', 'wb'), src_pts)
    # np.save(open('dst_pts.npy', 'wb'), dst_pts)
    norm_src_pts, norm_dst_pts, Ta, Tb = normalize(src_pts, dst_pts)
    assert norm_src_pts[0][0] != src_pts[0][0]
    
    print(src_pts.shape, len(matches))
    epsilon = 1e-3#c1e-3 o1e-3 p
    selected_pts = [norm_src_pts, norm_dst_pts]
    F = ransac_for_F(selected_pts, epsilon, Ta, Tb)
    
    # draw_epipolar_lines(img1,img2, src_pts, dst_pts, F)

    E = estimate_essential_mat(F, K)
    R_arr, T_arr = decompose_E(E)