import cv2

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\test3.mp4'

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# start_time = 57
# end_time = 125
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    # if current_time > end_time:
    #     break
    # text = str(int(current_time - 57))
    text = str(int(current_time))
    top_left = (10, 10)
    bottom_right = (80, 60)
    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), thickness=-1)

    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
