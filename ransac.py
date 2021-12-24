import random
import numpy as np
from normalized_8_point import fundamental_matrix, normalized_8_point

"""
RANSAC
Step 1: Choose Pairs
Step 2: Find Fundamental Matrix
Step 3: Calculate inliners against threshold
"""


def pair_chooser(points1, points2, size):
    """
    Choose 8 different Pairs of corresponding points
    :param points1: Points 1st
    :param points2: Points 2nd
    :param size: Total points
    :return: 8 pairs of corresponding points
    """
    indexes = random.sample(range(size), 8)
    return points1[indexes], points2[indexes]


def distance_pts_line(pts1, pts2, fundamental, iteration, threshold):
    """
    Finding distance X_p F X_q. if dist<threshold, mask gets 1 else 0
    :param pts1: Points 1
    :param pts2: Points 2
    :param fundamental: Fundamental Matrix
    :param iteration: Indexes
    :param threshold: Threshold
    :return:
    """
    return 1 if np.absolute(np.dot(pts1[iteration], np.dot(fundamental, pts2[iteration]))) < threshold else 0


def fundamental_ransac(points1, points2, prob_no_out=0.99, inliner_prob=0.5, threshold=2):
    """
    Find Fundamental Matrix and Mask using RANSAC algorithm
    :param points1: First View
    :param points2: Second View
    :param prob_no_out: Probability of sample with no outliners
    :param inliner_prob: Probability of observing an inliner
    :param threshold: Threshold for disregarding
    :return: Fundamental Matrix and Mask
    """
    # Step 1: Initialize Fundamental Matrix, mask and find optimal number of iterations
    # M = log(1 − p) / log(1 −v⁸)
    n = 0
    real_mask = None
    F = None
    M = int(np.log(1 - prob_no_out) / np.log(1 - inliner_prob ** 8))
    # M = 1000

    for i in range(M):
        # Step 2: Finding 8 pairs of points
        corresponding_pt1, corresponding_pt2 = pair_chooser(points1, points2, len(points1))

        # Step 3: Find Fundamental Matrix using 8 point Algorithm
        F_ = fundamental_matrix(corresponding_pt1, corresponding_pt2,
                                normalized_8_point(corresponding_pt1[:, 0], corresponding_pt1[:, 1]),
                                normalized_8_point(corresponding_pt2[:, 0], corresponding_pt2[:, 1]))

        # Step 4: Find number of inliners
        ans1 = np.concatenate((points1.T, np.ones([1, points1.shape[0]])), 0).T
        ans2 = np.concatenate((points2.T, np.ones([1, points2.shape[0]])), 0).T
        mask = [distance_pts_line(ans1, ans2, F_, j, threshold) for j in range(len(points1))]
        # TODO TWEAK THRESHOLD?
        inliners = sum(mask)

        # Step 5: If inliners is greater than previously found, update it along with mask and F
        if inliners > n:
            print(inliners)
            n = inliners
            F = F_
            real_mask = np.array(mask)

    # Step 6: Return Fundamental Matrix and Mask after end of iterations
    return F, real_mask.reshape((-1, 1))


# Original
# [[ 6.84469238e-06  1.52711820e-05 -4.55139862e-03]
#  [-1.64723597e-05  4.83390830e-06  4.45262204e-03]
#  [ 2.69635782e-04 -6.94241048e-03  1.00000000e+00]]

# My
# [[-3.00263243e-06  2.51971655e-05 - 8.20020396e-05]
#  [6.75309899e-05 - 8.49807369e-06 - 4.41270397e-02]
#  [-4.16016169e-03  2.13888670e-02  1.00000000e+00]]
