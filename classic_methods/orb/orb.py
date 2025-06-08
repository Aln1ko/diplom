import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


# Шлях до зображень
images = sorted(glob.glob("D:\\sasha\\4-course\\secondsemestr\\diplom\\dataset\\sequences\\10\\image_0\\*.png"))

filename = "D:\\sasha\\4-course\\secondsemestr\\diplom\\dataset\\sequences\\10\\calib.txt"
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
# print(P0)
K = P0[:, :3]
# print(K)

# файл с дійсними даними
filename_res = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\poses\\poses\\10.txt"

def create_results(filename_res):
    gt_data = []
    with open(filename_res, 'r') as f:
        for line in f:
            vals = list(map(float, line.split()))
            # Extract translation and rotation matrix
            R_gt = np.array([[vals[0], vals[1], vals[2]],
                             [vals[4], vals[5], vals[6]],
                             [vals[8], vals[9], vals[10]]])

            t_gt = np.array([[vals[3]], [vals[7]], [vals[11]]])
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

gt_abs = create_results(filename_res)
gt_delta = delta(gt_abs)


# Ініціалізація ORB та BFMatcher
orb = cv2.ORB_create(2000)
# orb = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Початкові значення
R_total = np.eye(3)
t_total = np.zeros((3, 1))
moving = []

preds_abs = []
preds_delta = []


for i in range(len(gt_abs)-1):
    if i %100 == 0:
        print(i)

    img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(images[i + 1], cv2.IMREAD_GRAYSCALE)

    # 1. Витяг ознак
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. Зіставлення
    # matches = bf.knnMatch(des1, des2, k = 2)
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    good_matches = bf.match(des1,des2)

    if len(good_matches) < 8:
        print(f"[{i}] Недостаточно матчей: {len(good_matches)}")
        continue

    # matches = sorted(matches, key=lambda x: x.distance)

    # 3. Отримання координат
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 4. Обчислення Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        print(f"[{i}] Помилка при обчисленні E")
        continue

     # --- Відбір тільки inliers ---
    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]

    if len(pts1_in) < 8:
        print(f"[{i}] Недостаточно inliers: {len(pts1_in)}")
        continue

    # 5. Відновлення поза (R, t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

    scale = np.linalg.norm(gt_delta[i][1])
    if scale < 0.1 or scale > 10:
        print(f"[{i}] Підозрілий масштаб: {scale:.2f}")
        continue
    scale = 1
    t = scale * t
    R = R.T
    t = -R @ t
    # создать массив, вносить єти р и т для рмсе и остального что в компут акураси

    preds_delta.append((R.copy(), t.copy()))
     # --- Масштаб із ground truth ---
    # R_gt_prev, t_gt_prev = gt[i]
    # R_gt_curr, t_gt_curr = gt[i + 1]
    # gt_delta11 = t_gt_curr - t_gt_prev
    # # gt_prev = np.array(gt [i])
    # # gt_curr = np.array(gt [i + 1])
    # # gt_delta = t_gt_curr - t_gt_prev
    # gts.append(gt[i+1])


    # scaled_t = scale * t
    # 6. Оновлення глобального положення
    # t_prev = R_total @ scaled_t
    # t_total = t_total + t_prev

    t_total = t_total + R_total @ t
    R_total = R_total @ R

    # x = t_total[0, 0]
    # y = t_total[1, 0]
    # z = t_total[2, 0]
    # res = [x,y,z]
    preds_abs.append((R_total.copy(), t_total.copy()))
    # if i % 0.1*int(len(gt)-1) == 0:
    #     print("1")
    # loss = criterion(res, gt )
    # (R_gt, t_gt) = gt [i]
    # print(f"[{i}] Location: x={x:.2f}, y={y:.2f}, z={z:.2f}| Ground truth:{t_gt} | scale = {scale}")

# print(preds_abs[200][1])
# print(gt_abs[200][1])
# print(np.linalg.norm(preds_abs[200][1]))
# print(np.linalg.norm(gt_abs[200][1]))

def compute_drift_over_distance(traj_pred, traj_gt, interval= 100):
    drift_list = []
    dist_traveled = 0.0

    for i in range(1, len(traj_gt)):
        step_dist = np.linalg.norm(traj_gt[i][1] - traj_gt[i - 1][1])
        dist_traveled += step_dist

        if dist_traveled >= interval:
            drift = np.linalg.norm(traj_gt[i][1] - traj_pred[i][1])
            drift_list.append(drift)
            dist_traveled = 0.0

    return drift_list

def classical_metrics_as_rmse(preds, gt, pos_tol= 0.1, ang_tol=0.02):
    n = len(preds_abs)
    sum_sq_pos = 0.0
    pos_corr = 0
    sum_angle = 0.0
    ang_corr = 0
    arr_err_t = []


    for (R_pred, t_pred), (R_gt, t_gt) in zip(preds, gt):
        # Position
        err_pos = np.linalg.norm(t_pred - t_gt)
        arr_err_t.append(err_pos)
        sum_sq_pos += err_pos**2
        # print(err_pos, '   ', sum_sq_pos)
        if err_pos <= pos_tol:
            pos_corr += 1

        # Rotation
        R_diff = R_pred @ R_gt.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        sum_angle += angle
        if angle <= ang_tol:
            ang_corr += 1

    rmse_pos = np.sqrt(sum_sq_pos / n)
    ate_mean = sum(arr_err_t) / n
    ate_median = np.median(arr_err_t)
    acc_pos = pos_corr / n
    avg_ang = sum_angle / n
    acc_ang = ang_corr / n

    return rmse_pos, ate_mean, ate_median, acc_pos, avg_ang, acc_ang



with open("orb_without_scale.txt", "w") as f:
    avg_loss, ate_mean, ate_median, pos_acc, avg_loss_angle, angle_acc = classical_metrics_as_rmse(preds_delta, gt_delta, 0.1, 0.1)
    print(' ')
    print(f"Rmse_local: {avg_loss:.3f} m")
    print(f"Position accuracy_local (@{0.1} m): {pos_acc:.3f}")
    print(f"Angle avg loss_local (@{0.1} rad): {avg_loss_angle:.3f}")
    print(f"Angle accuracy_local (@{0.1} rad): {angle_acc:.3f}")
    print(f"ATE mean_local: {ate_mean:.3f} m, median_local: {ate_median:.3f} m")
    print(' ')

    f.write("\n")
    f.write(f"Rmse_local: {avg_loss:.3f} m\n")
    f.write(f"Position accuracy_local (@{0.1} m): {pos_acc:.3f}\n")
    f.write(f"Angle avg loss_local (@{0.1} rad): {avg_loss_angle:.3f}\n")
    f.write(f"Angle accuracy_local (@{0.1} rad): {angle_acc:.3f}\n")
    f.write(f"ATE mean_local: {ate_mean:.3f} m, median_local: {ate_median:.3f} m\n")
    f.write("\n")

    drift = compute_drift_over_distance(preds_abs, gt_abs, interval= 100)
    avg_loss, ate_mean, ate_median, pos_acc, avg_loss_angle, angle_acc = classical_metrics_as_rmse(preds_abs, gt_abs, 5, 0.1)
    print(f"Rmse: {avg_loss:.3f} m")
    print(f"Position accuracy (@{5} m): {pos_acc:.3f}")
    print(f"Angle avg loss (@{0.1} rad): {avg_loss_angle:.3f}")
    print(f"Angle accuracy (@{0.1} rad): {angle_acc:.3f}")
    print(f"ATE mean: {ate_mean:.3f} m, median: {ate_median:.3f} m")
    rounded_data = [round(float(x), 4) for x in drift]
    print('Drift: ', rounded_data)
    # drift = compute_drift_over_distance(preds_abs, gt_abs, interval=100)
    # avg_loss, ate_mean, ate_median, pos_acc, avg_loss_angle, angle_acc = classical_metrics_as_rmse(preds_abs, gt_abs, 5, 0.1)

    f.write(f"Rmse: {avg_loss:.3f} m\n")
    f.write(f"Position accuracy (@{5} m): {pos_acc:.3f}\n")
    f.write(f"Angle avg loss (@{0.1} rad): {avg_loss_angle:.3f}\n")
    f.write(f"Angle accuracy (@{0.1} rad): {angle_acc:.3f}\n")
    f.write(f"ATE mean: {ate_mean:.3f} m, median: {ate_median:.3f} m\n")

    rounded_data = [round(float(x), 4) for x in drift]
    f.write("Drift: " + str(rounded_data) + "\n")



# print(' ')
# print(preds_delta[100][0])
# print(' ')
# print(gt_delta[100][0])
# print(' ')
# print(preds_delta[100][1])
# print(' ')
# print(gt_delta[100][1])
# print(' ')
# print(preds_abs[100][0])
# print(' ')
# print(gt_abs[100][0])
# print(' ')
# print(preds_abs[100][1])
# print(' ')
# print(gt_abs[100][1])

def plot_predicted_vs_true_trajectory(pred, true, save_path = "trajectory_plot_orb_without_scale.png"):
    # Собираем все t-позиции в массивы (N,3)
    pred_ts = np.vstack([t_pred.ravel() for R_pred, t_pred in pred])
    gt_ts   = np.vstack([t_gt.ravel()   for    R_gt,   t_gt   in true])

    # Берём столбцы x (0) и z (2)
    pred_xz = pred_ts[:, [0, 2]]
    gt_xz   = gt_ts  [:, [0, 2]]

    plt.figure(figsize=(8,6))
    # plt.plot(gt_ts[:,2], -gt_ts[:,0],
    plt.plot(gt_ts[:,0], gt_ts[:,2],
             '--', label="Ground Truth", linewidth=2)
    # plt.plot(pred_xz[:,0], pred_xz[:,1],
    plt.plot(pred_ts[:,0], pred_ts[:,2],
             '-',  label="Predicted",   linewidth=2)
    plt.scatter(gt_xz[0,0], gt_xz[0,1],
                c='red', marker='o', label="Start")
    plt.scatter(gt_xz[-1,0], gt_xz[-1,1],
                c='green', marker='x', label="End")

    plt.xlabel("X-позиція (м)")
    plt.ylabel("Z-позиція (м)")
    plt.title("Порівняння траєкторій (вид зверху)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Збереження графіка у файл
    plt.savefig(save_path)
    plt.show()

plot_predicted_vs_true_trajectory(preds_abs, gt_abs)
