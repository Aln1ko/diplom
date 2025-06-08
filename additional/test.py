import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size = (7,7), stride = (2,2), padding=(3, 3)),# 185, 613
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 185, 613
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),#93, 307
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 93, 307
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),#47, 154
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#47, 154
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),# 24, 77
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 24, 77
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),# 12, 39
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# 12, 39
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  #  6 × 20
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.AdaptiveMaxPool2d(3,10),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  #  3 × 10
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d((5, 5))  # Глобальний пулінг до фіксованого розміру
        )
        self.lstm1 = nn.LSTMCell(input_size = 512 * 3 * 10, hidden_size=128)
        # self.lstm2 = nn.LSTMCell(input_size = 512 * 3 * 10, hidden_size=128)
        self.lstm2 = nn.LSTMCell(input_size = 128, hidden_size=128)
        self.network = nn.Sequential(
            nn.Linear(128, 6)
        )
    def forward(self,image1,image2):
        images = torch.cat((image1,image2),dim = 1)

        out_features = self.features(images)
        out_features = torch.flatten(out_features, 1)
        batch_size = out_features.size(0)

        # Ініціалізуємо приховані стани та стани осередків для lstm1
        h_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)
        c_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)

        # Пропускаємо ознаки першої картинки через lstm1 (один часовий крок)
        h_t1_next, c_t1_next = self.lstm1(out_features, (h_t1, c_t1))

       # Пропускаємо ознаки першої картинки через lstm2 (один часовий крок)
        h_t2_next, c_t2_next = self.lstm2(h_t1_next, (h_t1_next, c_t1_next))

        out = self.network(h_t2_next) # [batch_size, 1]
        return out



transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((370,1226)),
        transforms.ToTensor()
    ])
model = Network()

path = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\10_epoch.pth"
checkpoint = torch.load(path,map_location=torch.device('cpu'))
model.load_state_dict( checkpoint['state_model'] )
model.eval()

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\reconstructed_video.mp4'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\fin_video1.mp4'
start_time = 0
end_time = 10
x1, y1, width, height = 150, 50, 1130, 325
speedo_x, speedo_y, speedo_w, speedo_h = 880, 440, 400, 280
overlay_x,overlay_y = width - speedo_w//3 ,height - speedo_h//3

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"fps: {fps}, width: {frame_width}, height:{frame_height}")
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Не удалось прочитать видео!")

# Конвертуємо в PIL та застосовуємо трансформації
prev_frame_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
prev_frame_tensor = transform(prev_frame_pil).unsqueeze(0) # [ 1, 1, H, W]
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps + 1)
i = 1
speed = 0  # Початкове значення
text = "0"  # рядок для відображення
frame_num = 1

step = 1
fps_int = int(fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    # cropped_frame = frame[y1:y1+height, x1:x1+width]
    cropped_frame = frame

    if frame_num % step == 0:
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor_image = transform(pil_image).unsqueeze(0)  # Результат: [1, 1, H, W]
    if i%fps == 0:
        with torch.no_grad():
            output = model(prev_frame_tensor, tensor_image)
        dx, dy, dz = output[0][0:3]  # перші три компоненти — це трансляція
        distance = torch.sqrt(dx**2 + dy**2 + dz**2).item()

        speed = distance*fps*3.6/ step
        print(f"distance is {distance}, speed is {speed}")
    if frame_num % step == 0:
        prev_frame_tensor = tensor_image
    i += 1
    frame_num += 1

    text = str(int(speed))
    top_left = (10, 10)
    bottom_right = (80, 60)
    cv2.rectangle(cropped_frame, top_left, bottom_right, (255, 255, 255), thickness=-1)

    cv2.putText(cropped_frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    # speddo = frame[speedo_y:speedo_y+speedo_h, speedo_x:speedo_x+speedo_w]
    # speddo_small = cv2.resize(speddo, (speedo_w // 3, speedo_h // 3))
    # h, w, _ = speddo_small.shape

    # cropped_frame[overlay_y:overlay_y+h, overlay_x:overlay_x+w] = speddo_small


    if current_time > end_time:
        break
    out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
