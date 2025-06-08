import cv2
import os
import glob

# Папка з кадрами
input_folder = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\dataset\\sequences\\10\\image_0'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\reconstructed_video.mp4'

# Отримуємо список зображень, відсортований за іменем
image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))

# Перевірка: чи є зображення
if not image_files:
    raise ValueError(f"Не знайдено жодного зображення у папці: {input_folder}")

# Зчитуємо розмір першого зображення
frame = cv2.imread(image_files[0])
height, width, _ = frame.shape

# Встановіть бажану частоту кадрів (fps)
fps = 10  # або використайте те ж саме, що й у оригінальному відео

# Ініціалізуємо запис відео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # або 'XVID' для .avi
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Додаємо кожен кадр
for image_file in image_files:
    frame = cv2.imread(image_file)
    out.write(frame)

# Завершуємо запис
out.release()
cv2.destroyAllWindows()

print(f'Відео збережено у файл: {output_video_path}')
