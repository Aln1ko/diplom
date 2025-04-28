import cv2
video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
out_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\test2.mp4'
cap = cv2.VideoCapture(video_path)
x1, y1, width, height = 150, 50, 1130, 325
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret: break
    cropped_frame = frame[y1:y1+height, x1:x1+width]
    out.write(cropped_frame)
cap.release()
out.release()
cv2.destroyAllWindows()
print("done")
