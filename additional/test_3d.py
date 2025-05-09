
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 5, 180, 320)
            out = self._forward_conv(dummy)
            self.flatten_size = out.view(1, -1).shape[1]
        # Output size after convs: (B, 64, 1, 46, 153)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 1)  # [x, y, z, roll, pitch, yaw]

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((180, 320)),
        # transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
model = Network()

path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\code1\\additional\\7_epoch.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict( checkpoint['state_model'] )
model.eval()

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\fin_video.mp4'

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


start_time = 57
# end_time = 87
video_duration = 30
# step = 1

speed = 0 # Початкове значення
text = "0" # рядок для відображення
frame_num = 0
max_frame_num = video_duration * fps
fps_int = int(fps)
frames_list = []
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

while (frame_num < max_frame_num):
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = frame[y1:y1+height, x1:x1+width]

    if frame_num % fps_int < 5:
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        # Перетворення numpy array на PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Застосування трансформацій
        frames_list.append(transform(pil_image))  # Результат: [1, H, W]

        if frame_num % fps_int == 4:
            frames_list = torch.stack(frames_list, dim=0)  # -> [5, 1, H, W]
            frames_list = frames_list.permute(1, 0, 2, 3).unsqueeze(0) # -> [1, 5, H, W] -> [1, 1, 5, H, W]
            with torch.no_grad():
                output = model(frames_list)
            distance = output[0][0].item()
            # speed = distance*fps*3.6/ step
            speed = distance*fps*3.6
            frames_list = []

            print(f"distance is {distance}, speed is {speed}")

    frame_num += 1

    text = str(int(speed))
    top_left = (10, 10)
    bottom_right = (80, 60)
    cv2.rectangle(cropped_frame, top_left, bottom_right, (255, 255, 255), thickness=-1)

    cv2.putText(cropped_frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    speddo = frame[speedo_y:speedo_y+speedo_h, speedo_x:speedo_x+speedo_w]
    speddo_small = cv2.resize(speddo, (speedo_w // 3, speedo_h // 3))
    h, w, _ = speddo_small.shape

    cropped_frame[overlay_y:overlay_y+h, overlay_x:overlay_x+w] = speddo_small

    out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
