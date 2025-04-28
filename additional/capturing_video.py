import cv2

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\cropped_video2.mp4'

start_time = 57
end_time = 125

x1, y1, width, height = 150, 50, 1130, 325
speedo_x, speedo_y, speedo_w, speedo_h = 880, 440, 400, 280
overlay_x,overlay_y = width - speedo_w//3 ,height - speedo_h//3

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"fps: {fps}, width: {frame_width}, height:{frame_height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

while True:
    ret, frame = cap.read()
    if not ret :
        break

    current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    if current_time > end_time:
        break

    cropped_frame = frame[y1:y1+height, x1:x1+width]

    # speddo = frame[speedo_y:speedo_y+speedo_h, speedo_x:speedo_x+speedo_w]
    # speddo_small = cv2.resize(speddo, (speedo_w // 3, speedo_h // 3))
    # h, w, _ = speddo_small.shape

    # cropped_frame[overlay_y:overlay_y+h, overlay_x:overlay_x+w] = speddo_small
    out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("done")



# import cv2

# video_path = 'path_to_video.mp4'
# out_video_path = 'path_to_out_video.mp4'
# cap = cv2.VideoCapture(video_path)

# start_time = 57
# end_time = 125

# x1, y1, width, height = 150, 50, 1130, 325

# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
#     if current_time > end_time:
#         break

#     cropped_frame = frame[y1:y1+height, x1:x1+width]
#     out.write(cropped_frame)

# cap.release()
