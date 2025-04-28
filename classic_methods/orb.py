import cv2
import numpy as np
import glob

# Шлях до зображень (заміни на свій)
images = sorted(glob.glob("D:\\sasha\\4-course\\secondsemestr\\diplom\\dataset\\sequences\\09\\image_0\\*.png"))  # наприклад, ./data/00/image_0/*.png

filename = "D:\\sasha\\4-course\\secondsemestr\\diplom\\dataset\\sequences\\09\\calib.txt"
def read_p0_from_calib_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                # Відокремлюємо числові значення після 'P0:'
                values = list(map(float, line.strip().split()[1:]))
                # Перетворюємо в 3x4 numpy матрицю
                P0 = np.array(values).reshape(3, 4)
                return P0
    raise ValueError("P0 не знайдена у файлі")

P0 = read_p0_from_calib_file(filename)
print(P0)
K = P0[:, :3]
print(K)

# файл с настоящими данными
filename_res = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\09.txt"
def create_results(filename_res):
    lines = []
    with open(filename_res, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            lines.append(data)
    res = []
    for i in range(len(lines) - 1):
        result = f" x = {float(lines[i+1][3]):.2f}, y = {float(lines[i+1][7]):.2f}, z = {float(lines[i+1][11]):.2f}"
        res.append(result)
    return res


res = create_results(filename_res)
# Ініціалізація ORB та BFMatcher
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Початкові значення
R_total = np.eye(3)
t_total = np.zeros((3, 1))
moving = []

# for i in range(len(images)-1):
for i in range(100):
    img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)

    # 1. Витяг ознак
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. Зіставлення
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. Отримання координат
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 4. Обчислення Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # 5. Відновлення поза (R, t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # 6. Оновлення глобального положення
    # print(f"[{i}] t: {t.flatten()}")
    # s = R_total @ t
    # t_total += s
    t_total += R_total @ t
    R_total = R @ R_total
    # moving.append( (s[0, 0] ** 2 + s[1, 0] ** 2 + s[2, 0] ** 2) ** 0.5)
    # print(f"[{i}] Norm of s: {moving[-1]}")

    print(f"[{i}] Location: x={t_total[0, 0]:.2f}, y={t_total[1, 0]:.2f}, z={t_total[2, 0]:.2f}| Ground truth:{res[i]}")


# movements_file = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\09.txt"
# def create_movements(movements_file):
#         movements = []
#         with open(movements_file, 'r') as f:
#             for line in f:
#                 data = list(map(float, line.strip().split()))
#                 movements.append(data)
#         return movements

# mov = create_movements(movements_file)

# def transform_mov(mov):
#         res_movements = []
#         for i in range(len(mov) - 1):
#         # for i in range(100):
#             res = ( (mov[i+1][3] - mov[i][3]) ** 2 + (mov[i+1][7] - mov[i][7]) ** 2 + (mov[i+1][11] - mov[i][11]) ** 2 ) ** 0.5
#             res_movements.append(res)
#         return res_movements

# mov_1 = transform_mov(mov)
# print(len(moving), '  ', len(mov_1))
# for i in range(100):
#     print(moving[i], '   ', mov_1[i])
