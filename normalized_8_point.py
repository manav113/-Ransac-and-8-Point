import numpy as np

"""
NORMALIZE INPUT
1. Find the centroid of the points (find the mean x and mean y value)
2. Compute the mean distance of all the points from this centroid
3. Construct a 3 by 3 matrix that would translate the points so that the mean distance
would be sqrt(2)
"""


def centroid_points(x, y):
    """
    Finding centroid of the points
    :param x: X value
    :param y: Y value
    :return: Mean
    """
    return sum(x) / x.shape[0], sum(y) / y.shape[0]


def distance(x, cent_x, y, cent_y):
    """
    Compute Mean Distance from centroid
    :param x: X values
    :param cent_x: X Centroid
    :param y: Y Values
    :param cent_y: Y Centroid
    :return: Mean distance
    """
    return np.mean(np.sqrt((x - cent_x) ** 2 + (y - cent_y) ** 2))


def components(scale, x_comp, y_comp):
    """
    Making the distance sqrt(2) ( Isotropic Normalization?)
    :param scale: Distance
    :param x_comp: Centroid X
    :param y_comp: Centroid Y
    :return: Scale Factor, Scaled_Centroid
    """
    s = np.sqrt(2) / scale
    return s, s * x_comp, s * y_comp


def normalized_8_point(x, y):
    """
    Responsible for normalizing the points
    :param x: X points
    :param y: Y points
    :return: Transformation Matrix
    """
    # Step 1: Finding centroid of the points
    cent_x, cent_y = centroid_points(x, y)

    # Step 2: Getting Mean Distance
    res = distance(x, cent_x, y, cent_y)

    # Step 3: Scale so that mean distance would be sqrt(2)
    res, x_scaled, y_scaled = components(res, cent_x, cent_y)

    return translate_matrix(res, x_scaled, y_scaled)


def translate_matrix(res, x_scaled, y_scaled):
    """
    Getting the Transformation Matrix
    T = | Ax    0   Dx  |
        | 0     Ay  Dy  |
        | 0     0   1   |
    :param res: Ax, Ay
    :param x_scaled: Dx
    :param y_scaled: Dy
    :return: Transformation Matrix
    """
    return np.array([[res, 0, -x_scaled],
                     [0, res, -y_scaled],
                     [0, 0, 1]])


"""
Fundamental Matrix
Step 1: Find Matrix A
Step 2: Find Linear Solution
Step 3: Singularity Constraint
Step 4: Un-Normalize
"""


def dot_product(pts, matrix):
    """
    Normalize the points by multiplying with Dot Product
    X_norm = T*x
    :param pts: Points to Normalize
    :param matrix: Transformation Matrix (refer to translate_matrix func)
    :return: Normalized Points
    """
    # Step 1: Appending 1 to make it homogenous
    ans1 = np.concatenate((pts.T, np.ones([1, pts.shape[0]])), 0)

    # Step 2: Return the dot product of the points with the Transformation Matrix
    return matrix.dot(ans1)


def equation(length, pts1, pts2):
    """
     Finding the Matrix A so that we can solve for Af = 0
     where A =  [xl_xr xl_yr xl  yl_xr  yl_yr  yl  xr yr 1]
                    :   :    :     :       :    :  :  :  :
                [xl_xr xl_yr xl  yl_xr  yl_yr  yl  xr yr 1]

     and f = [f1 f2 f3 f4 f5 f6 f7 f8 f9]'
    :param length: No. of points
    :param pts1: Points from 1st View Normalized
    :param pts2: Points from 2nd View Normalized
    :return: Coefficient Matrix
    """
    # Step 1: Make a zeros A Matrix and then populate it with the coefficients
    A = np.zeros((length, 9))
    # Step 2: Add the coefficients and return
    for i in range(length):
        xix_ = pts1[0, i] * pts2[0, i]  # xl_xr
        xiy_ = pts1[0, i] * pts2[1, i]  # xl_yr
        xi = pts1[0, i]  # xl
        yix_ = pts1[1, i] * pts2[0, i]  # yl_xr
        yiy_ = pts1[1, i] * pts2[1, i]  # yl_yr
        yi = pts1[1, i]  # yl
        x_ = pts2[0, i]  # xr
        y_ = pts2[1, i]  # yr
        A[i] = [xix_, xiy_, xi,
                yix_, yiy_, yi,
                x_, y_, 1
                ]

    return A


def svd(matrix):
    """
    Finding SVD Of the Matrix and enforce Rank = 2 by setting singular value to be zero
    :param matrix: Matrix A
    :return: SVD Decomposition of A
    """
    # Step 1: Finding SVD of A ( A = USV )
    _, _, v = np.linalg.svd(matrix)
    # Step 2: V is a 9x9 Matrix. The last column has lowest singular value, thus using the last column
    v = v.T[:, -1].reshape((3, 3))
    U, S, V = np.linalg.svd(v.T)
    # Step 3: Enforce F to be rank 2 and return U,S,V
    S[2] = 0
    return U, S, V


def un_normalize(matrix, pts1, pts2):
    """
    De-Normalize F ( F(dem) = T_r F T_l )
    :param matrix: Fundamental Matrix
    :param pts1: Transformation Matrix from 1st View
    :param pts2: Transformation Matrix from 2nd View
    :return: Un-normalized F
    """
    F = np.dot(pts2.T, matrix.dot(pts1))
    return F / F[2, 2]


def fundamental_matrix(pts1, pts2, normalized_pts1, normalized_pts2):
    """
    Find Fundamental Matrix using 8 point Algorithm
    :param pts1: Points from 1st View
    :param pts2: Points from 2nd View
    :param normalized_pts1: Transformation Matrix from 1st view
    :param normalized_pts2: Transformation Matrix from 2nd view
    :return: Fundamental Matrix
    """
    # Step 1: Find the normalized points wrt to the matrix
    x = dot_product(pts1, normalized_pts1)
    y = dot_product(pts2, normalized_pts2)

    # Step 2: Find the Matrix A
    matrix_A = equation(len(pts1), x, y)

    # Step 3: Find the Fundamental Matrix and Enforce Rank = 2
    U, S, V = svd(matrix_A)
    F = np.dot(U, np.dot(np.diag(S), V))

    # Step 4: Return un_normalized Fundamental Matrix
    return un_normalize(F, normalized_pts1, normalized_pts2)

# Original Left / Right
# [[ 1.30650775e-05  1.81420944e-05 -3.61564062e-03]
# [-2.57558327e-05  1.90938190e-05  2.13164449e-03]
# [-1.84707041e-03 -7.93245689e-03  1.00000000e+00]]

# My
# [[ 1.30650775e-05  1.81420944e-05 -3.61564062e-03]
# [-2.57558327e-05  1.90938190e-05  2.13164449e-03]
# [-1.84707041e-03 -7.93245689e-03  1.00000000e+00]]
