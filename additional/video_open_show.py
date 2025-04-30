import cv2

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Помилка при відкритті відео")
else:
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()


# import cv2
# video_path = 'path_to_your_video.mp4'
# cap = cv2.VideoCapture(video_path)

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break
# cap.release()
