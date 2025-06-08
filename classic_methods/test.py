import numpy as np
import math

movemant_file = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\09.txt"

def create_movements(movements_file):
        movements = []
        with open(movements_file, 'r') as f:
            for line in f:
                data = list(map(float, line.strip().split()))
                movements.append(data)
        return np.array(movements)

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def get_relative_pose(pose1, pose2,):
    if len(pose1) == 12:
        pose1 = np.concatenate((pose1.reshape(3, 4), np.array([[0, 0, 0, 1]])), axis=0)
    if len(pose2) == 12:
        pose2 = np.concatenate((pose2.reshape(3, 4), np.array([[0, 0, 0, 1]])), axis=0)

    rel_pose = np.dot(np.linalg.inv(pose1), pose2)

    dx = rel_pose[0, 3]
    dy = rel_pose[1, 3]
    dz = rel_pose[2, 3]

    rel_rot_matrix = rel_pose[:3, :3]
    rx, ry, rz = rotationMatrixToEulerAngles(rel_rot_matrix)

    return dx, dy, dz, rx, ry, rz

def transform_mov(movements):
        res_movements = []
        for i in range(len(movements) -1):
            m1 = movements[i]
            m2 = movements[i + 1]
            dx, dy, dz, rx, ry, rz = get_relative_pose(m1, m2)
            res_movements.append([dx, dy, dz, rx, ry, rz])
        return res_movements


mov = create_movements(movemant_file)
mov = transform_mov(mov)
print(mov[5])


filename_res = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\09.txt"

def create_results(filename_res):
    gt_data = []
    with open(filename_res, 'r') as f:
        for line in f:
            vals = list(map(float, line.split()))
            # Extract translation and rotation matrix
            R_gt = np.array([[vals[0], vals[1], vals[2]],
                             [vals[4], vals[5], vals[6]],
                             [vals[8], vals[9], vals[10]]])
            # R_to = np.array([[0, -1, 0],
            #                  [0, 0, -1],
            #                  [1, 0, 0]])
            # R_gt = R_gt @ R_to

            t_gt = np.array([vals[3], vals[7], vals[11]])
            # t_gt = np.array([vals[11], -vals[3], -vals[7]])

            # R_gt = R_gt.T
            # t_gt = -R_gt @ t_gt
            gt_data.append((R_gt, t_gt))
    return gt_data


def delta(gt_data):
    arr_delta = []
    for i in range(len(gt_data) - 1):
        pose1_R, pose1_t = gt_data[i]
        pose2_R, pose2_t = gt_data[i + 1]
        delta_R = pose1_R.T @ pose2_R
        delta_t = pose1_R.T @ (pose2_t - pose1_t)
        arr_delta.append((delta_R, delta_t))
    return arr_delta

gt = create_results(filename_res)
gt_delta = delta(gt)

print(gt_delta[5])
