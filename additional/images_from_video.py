import cv2
import os

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
output_folder = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

start_time = 57
end_time = 125

fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    if current_time > end_time:
        break

    filename = os.path.join(output_folder, f'frame_{frame_num:05d}.png')
    cv2.imwrite(filename, frame)
    frame_num += 1

cap.release()
cv2.destroyAllWindows()

print(f'Збережено {frame_num} кадрів в папку "{output_folder}"')
