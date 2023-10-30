from models import UNet,UNetModel
import torch
from torchvision.io import read_image, write_jpeg
import torchvision.transforms.functional as F 
from torchvision.transforms import InterpolationMode
import cv2
import pandas as pd
import argparse
import time
import tqdm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--Source","-s",type=str,help="Specify path of images to run inference on")
    
    ap.add_argument("--Destination", "-d",type=str,help="Specify path to save output images")
    args = vars(ap.parse_args())

    file_path = args["Source"]
    dst_path = args["Destination"]

    test_files = pd.read_csv(file_path)["image"]
    total_time = 0

    for file in tqdm.tqdm(test_files):
        start_time = time.perf_counter()
        img = read_image("D:\\Deep_Learning\\Segmentation\\Forest Segmentation\\Forest_Segmented\\images\\"+file)

        img = img.float()

        img = F.resize(img,size=(128))

        img = torch.unsqueeze(img,0)

        img = img / 255.0
        
        model = UNet()

        saved_dict = torch.load("model.pth",map_location='cuda')


        model.load_state_dict(saved_dict)

        y_hat = model(img).detach()

        y_hat[y_hat >= 0.5] = 255
        y_hat[y_hat < 0.5] = 0

        y_hat = torch.squeeze(y_hat,dim=0)
        y_hat = y_hat.to(torch.uint8)
        
        write_jpeg(y_hat,dst_path+"/Pred_"+file)

        end_time = time.perf_counter()

        total_time += (end_time - start_time)

        

    print(f"Time taken {total_time/len(test_files)} s")