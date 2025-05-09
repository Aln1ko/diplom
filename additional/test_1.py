
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Завантажуємо переднавчену модель ResNet50
#         weights = ResNet50_Weights.DEFAULT
#         base_model = resnet50(weights=weights)

#         # Міняжмо перший conv слой для 1-канального зображенняЗ
#         self.backbone = nn.Sequential()
#         conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Копіюємо та усереднюємо ваги по каналам
#         with torch.no_grad():
#             conv1.weight[:] = base_model.conv1.weight.mean(dim=1, keepdim=True)
#         self.backbone.add_module("conv1", conv1)

#         #Додаємо інші блоки ResNet до avgpool (виключая fc)
#         for name, module in list(base_model.named_children())[1:-2]:
#             self.backbone.add_module(name, module)

#         self.lstm1 = nn.LSTMCell(input_size = 2048*1*3, hidden_size=128)
#         self.lstm2 = nn.LSTMCell(input_size = 128, hidden_size=128)

#         self.classifier = nn.Linear(128, 1)

#     def forward(self,image1,image2):
#           images = torch.cat((image1,image2),dim = 1)

#           out_features = self.backbone(images)
#           out_features = torch.nn.functional.adaptive_avg_pool2d(out_features, (1,3 ))  # [B, C, 3, 10]
#           out_features = out_features.view(out_features.size(0), -1)  # [B, C * 3 * 10]

#         #   out_features = torch.flatten(out_features, 1)
#           batch_size = out_features.size(0)

#           # Ініціалізуємо приховані стани та стани осередків для lstm1
#           h_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)
#           c_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)

#           # Пропускаємо ознаки першої картинки через lstm1 (один часовий крок)
#           h_t1_next, c_t1_next = self.lstm1(out_features, (h_t1, c_t1))

#         # Пропускаємо ознаки першої картинки через lstm2 (один часовий крок)
#           h_t2_next, c_t2_next = self.lstm2(h_t1_next, (h_t1_next, c_t1_next))

#           out = self.classifier(h_t2_next) # [batch_size, 1]
#           return out



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 2, out_channels= 32, kernel_size= 10, stride= 2) # 86, 156
        self.bn1 =   nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 10, stride= 2, groups= 32) # 39, 74
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 1)

        self.conv4 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 5, stride= 2) # 18, 35
        self.bn2 =   nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, stride= 1, groups= 64) # 16, 33
        self.conv6 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 1)

        self.conv7 = nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 4, stride= 2) # 7, 15
        self.bn3 =   nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3, stride= 1, groups= 128) # 5, 13
        self.conv9 = nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 1)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.conv10 = nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, stride= 1) # 3, 11
        self.bn4 =   nn.BatchNorm2d(256)
        #self.conv8 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 1, groups= 32) #
        #self.conv9 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 1)

        self.lin1 = nn.Linear(256 * 3 * 11, 24)
        self.dropout = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(24, 1)
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, image1, image2):
        x = torch.cat([image1, image2], dim= 1)

        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu6(x)

        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.bn1(x)
        x = self.relu6(x)

        x = self.conv4(x)
        #x = self.bn2(x)
        x = self.relu6(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #x = self.bn2(x)
        x = self.relu6(x)
        #x = self.dropout2d(x)

        x = self.conv7(x)
        #x = self.bn3(x)
        x = self.relu6(x)
        x = self.conv8(x)
        x = self.conv9(x)
        #x = self.bn3(x)
        x = self.relu6(x)

        x = self.conv10(x)
        #x = self.bn4(x)
        x = self.relu6(x)

        x = torch.flatten(x, start_dim= 1)
        x = self.lin1(x)
        x = self.relu(x)
        # print(x)
        x = self.dropout(x)
        out = self.lin2(x)

        return  out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # nn.init.uniform_(m.weight, -1.0, 1.0)
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)

transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((180, 320)),
        # transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])

model = Network()

# path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\code1\\additional\\1_epoch.pth'
path = "D:\\sasha\\4-course\\secondsemestr\\diplom\\доп инфа\\model_5_0-20250505T050356Z-1-001\\model_5_0\\20_epoch.pth"
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
step = 1

speed = 0 # Початкове значення
text = "0" # рядок для відображення
frame_num = 0
max_frame_num = video_duration * fps
fps_int = int(fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

while (frame_num < max_frame_num):
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = frame[y1:y1+height, x1:x1+width]
    #Конвертація BGR (OpenCV) у RGB
    if frame_num % fps_int == 0:
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        # Перетворення numpy array на PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Застосування трансформацій
        prev_tensor_image = transform(pil_image).unsqueeze(0)  # Результат: [1, 1, H, W]
    if frame_num % fps_int == step:
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        # Перетворення numpy array на PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Застосування трансформацій
        tensor_image = transform(pil_image).unsqueeze(0)  # Результат: [1, 1, H, W]
        with torch.no_grad():
            output = model(prev_tensor_image, tensor_image)
        distance = output[0][0].item()
        speed = distance*fps*3.6/ step
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
