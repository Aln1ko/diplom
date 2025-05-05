import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Завантажуємо переднавчену модель ResNet50
        weights = ResNet50_Weights.DEFAULT
        base_model = resnet50(weights=weights)

        # Міняжмо перший conv слой для 1-канального зображенняЗ
        self.backbone = nn.Sequential()
        conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Копіюємо та усереднюємо ваги по каналам
        with torch.no_grad():
            conv1.weight[:] = base_model.conv1.weight.mean(dim=1, keepdim=True)
        self.backbone.add_module("conv1", conv1)

        #Додаємо інші блоки ResNet до avgpool (виключая fc)
        for name, module in list(base_model.named_children())[1:-2]:
            self.backbone.add_module(name, module)

        self.lstm1 = nn.LSTMCell(input_size = 2048*1*3, hidden_size=128)
        self.lstm2 = nn.LSTMCell(input_size = 128, hidden_size=128)

        self.classifier = nn.Linear(128, 1)

    def forward(self,image1,image2):
          images = torch.cat((image1,image2),dim = 1)

          out_features = self.backbone(images)
          out_features = torch.nn.functional.adaptive_avg_pool2d(out_features, (1,3 ))  # [B, C, 3, 10]
          out_features = out_features.view(out_features.size(0), -1)  # [B, C * 3 * 10]

        #   out_features = torch.flatten(out_features, 1)
          batch_size = out_features.size(0)

          # Ініціалізуємо приховані стани та стани осередків для lstm1
          h_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)
          c_t1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=out_features.device)

          # Пропускаємо ознаки першої картинки через lstm1 (один часовий крок)
          h_t1_next, c_t1_next = self.lstm1(out_features, (h_t1, c_t1))

        # Пропускаємо ознаки першої картинки через lstm2 (один часовий крок)
          h_t2_next, c_t2_next = self.lstm2(h_t1_next, (h_t1_next, c_t1_next))

          out = self.classifier(h_t2_next) # [batch_size, 1]
          return out

transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
model = Network()

path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\code\\additional\\1_epoch.pth'
checkpoint = torch.load(path,map_location=torch.device('cpu'))
model.load_state_dict( checkpoint['state_model'] )
model.eval()

video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\start_video\\video_test.mp4'
output_video_path = 'D:\\sasha\\4-course\\secondsemestr\\diplom\\test\\fin_video\\fin_video.mp4'
start_time = 57
end_time =60

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
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Не удалось прочитать видео!")

# Конвертируем в PIL и применяем трансформации
prev_frame_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
prev_frame_tensor = transform(prev_frame_pil).unsqueeze(0) # [ 1, 1, H, W]
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps+1)
i = 1
speed = 0  # начальное значение
text = "0"  # строка для отображения
frame_num = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
    print(current_time)
    cropped_frame = frame[y1:y1+height, x1:x1+width]
    # Конвертация BGR (OpenCV) в RGB

    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    # Преобразование numpy array в PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # Применение трансформаций
    tensor_image = transform(pil_image).unsqueeze(0)  # Результат: [1, 1, H, W]
    if i%fps == 0:
        with torch.no_grad():
            output = model(prev_frame_tensor, tensor_image)
        distance = output[0][0].item()
        speed = distance*fps*3.6/6
    if frame_num%6 == 0:
        prev_frame_tensor = tensor_image
    i +=1
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


    if current_time > end_time:
        break
    # cv2.imshow('Video', cropped_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    out.write(cropped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
