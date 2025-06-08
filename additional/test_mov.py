import numpy as np
import math
from scipy.spatial.transform import Rotation as R

path = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\00.txt"

def create_movements(movements_file):
        movements = []
        with open(movements_file, 'r') as f:
            for line in f:
                data = list(map(float, line.strip().split()))
                movements.append(data)
        return np.array(movements)

mov = create_movements(path)
print(mov[:2])

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

mov = transform_mov(mov)
print(mov[:2])

def cumulative_trajectory(relative_coords, start_pose= np.eye(4)):
    absolute_poses = [start_pose]
    current_pose = start_pose.copy()
    for delta in relative_coords:
        dx, dy, dz, rx, ry, rz = delta

        # Створення матриці обертання з кутів Ейлера (в радіанах)
        rotation = R.from_euler('xyz', [rx, ry, rz]).as_matrix()

        # Створення матриці трансляції
        translation = np.array([[1, 0, 0, dx],
                                [0, 1, 0, dy],
                                [0, 0, 1, dz],
                                [0, 0, 0, 1]])

        # Створення матриці відносної пози
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = rotation
        relative_pose[:3, 3] = [dx, dy, dz]

        # Оновлення поточної абсолютної пози шляхом множення на відносну
        current_pose = np.dot(current_pose, relative_pose)
        absolute_poses.append(current_pose)

    return np.array(absolute_poses[1:])  # пропускаємо початкову позу

mov = cumulative_trajectory(mov)
print(mov[:2])
