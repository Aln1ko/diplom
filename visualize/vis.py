import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = torch.randn(1, 10)
y = model(x)
writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.randn(1, 10))
writer.close()

# Визуализация графа
# make_dot(y, params=dict(model.named_parameters())).render("model_graph", format="png")
