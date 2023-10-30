import torch.onnx
from models import UNet

model = UNet()

model.load_state_dict(torch.load("model.pth"))

model.eval()

dummy_input = torch.randn((1,3,128,128))

torch.onnx.export(model,dummy_input,"forest_unet.onnx",verbose=True)
