import cv2
import numpy as np
import random
import time


def knn(descriptors_1, descriptors_2, neighbours=2):
    # Query Index, Train Index, Distance
    matches = []
    for i in range(0, len(descriptors_1)):
        dists = [tuple((j, np.linalg.norm(descriptors_1[i] - descriptors_2[j]))) for j in range(0, len(descriptors_2))]
        dists.sort(key=lambda d: d[1])
        matches.append([tuple((i, dists[k][0], dists[k][1])) for k in range(0, neighbours)])

    matches = [(m[0][1], m[0][0]) for m in matches if len(m) == neighbours and m[0][2] < m[1][2] * 0.75]
    return matches


def build_h_equations(imagePoints1, imagePoints2):
    h_equations = []
    for i in range(0, len(imagePoints1)):
        x1, y1 = imagePoints1[i][0], imagePoints1[i][1]
        x2, y2 = imagePoints2[i][0], imagePoints2[i][1]
        h_equations.append(np.array([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]))
        h_equations.append(np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]))

    return np.array(h_equations)


def get_homography_matrix(h_equations):
    u, e, v = np.linalg.svd(np.dot(h_equations.transpose(), h_equations))
    h = v[-1]
    hm = h / np.linalg.norm(h)
    hm /= h[-1]
    return hm.reshape(3, 3)


def ransac(imagePts1, imagePts2):
    homography_mat = None
    new_col = np.ones((imagePts1.shape[0], 1))
    homogeneous_pts1 = np.append(imagePts1, new_col, axis=1).transpose()
    indices = np.arange(len(imagePts1))
    iterations, inlier_count = 0, 0
    imageShape = imagePts1.shape[0]

    while iterations < 5:
        np.random.seed(29)
        indices_to_sample = np.random.choice(indices, 4)

        h_equations = build_h_equations(imagePts1[indices_to_sample], imagePts2[indices_to_sample])
        hm = get_homography_matrix(h_equations)

        hPi = np.dot(hm, homogeneous_pts1).transpose()
        norm_hPi = hPi / hPi[:, -1][:, None]
        norm_hPi = norm_hPi[:, :-1]

        inlier_indices = np.linalg.norm(imagePts2 - norm_hPi, axis=1) < 5
        curr_count = np.count_nonzero(inlier_indices)
        if curr_count > inlier_count:
            homography_mat = hm
            inlier_count = curr_count

        if curr_count > 0.45 * imageShape:
            break
        iterations += 1

    return homography_mat


def get_min(c_left_img, c_right_img):
    return np.sort(np.ravel((*np.min(c_left_img, axis=0), *np.min(c_right_img, axis=0))))[:2]


def get_max(c_left_img, c_right_img):
    return np.sort(np.ravel((*np.max(c_left_img, axis=0), *np.max(c_right_img, axis=0))))[::-1][:2]


def warp_perspective(left_img, right_img, homography_mat):
    height_left, width_left = left_img.shape[:2]
    height_right, width_right = right_img.shape[:2]

    corners_left_img = np.array([[[0, 0]], [[0, height_left]], [[width_left, height_left]],
                                 [[width_left, 0]]], dtype=np.float32)
    corners_right_img = np.array([[[0, 0]], [[0, height_right]], [[width_right, height_right]],
                                  [[width_right, 0]]], dtype=np.float32)

    corners_right_img = cv2.perspectiveTransform(corners_right_img, homography_mat)

    points_min = -get_min(corners_left_img, corners_right_img).astype(int)
    points_max = get_max(corners_left_img, corners_right_img).astype(int)

    offset_mask = np.array([[1, 0, points_min[0] + 0.5], [0, 1, points_min[1] - 1], [0, 0, 1]])
    homography_mat = offset_mask.dot(homography_mat)

    new_image_points = (points_min[0] + points_max[0], points_min[1] + points_max[1])
    result = cv2.warpPerspective(left_img, homography_mat, new_image_points)
    result[points_min[1]:height_left + points_min[1], points_min[0]:width_left + points_min[0]] = right_img
    return result


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: Return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    start_time = time.time()
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)

    keypoints_1, descriptors_1 = sift.detectAndCompute(left_img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(right_img, None)

    matches = knn(descriptors_1, descriptors_2)

    image_points1, image_points2 = [], []
    if len(matches) > 4:
        image_points1 = np.array([keypoints_1[i].pt for (_, i) in matches])
        image_points2 = np.array([keypoints_2[i].pt for (i, _) in matches])

    homography_mat = ransac(image_points1, image_points2)

    result = warp_perspective(left_img, right_img, homography_mat)
    print(f'Total Time taken: {time.time() - start_time}')
    return result


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/stitching_result.jpg', result_img)
