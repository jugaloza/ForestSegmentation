from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision.io import read_image, ImageReadMode, write_jpeg
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import lightning.pytorch as pl

class GenerateData(Dataset):
    def __init__(self,isTrain=True, train_split=0.8) -> None:
        super().__init__()

        self.files = pd.read_csv("meta_data.csv")
        #self.transform = Compose([Resize(size=(128,128)),Normalize(mean=[0.34,0.33,0.23],std=[0.10,0.08,0.07])])
        self.maskTransform = Resize(size=(128,128))
        self.imageTransform = Resize(size=(128,128))
        


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        
        img = read_image("Forest_Segmented/images/" + self.files.loc[index]['image'])
        mask = read_image("Forest_Segmented/masks/" + self.files.loc[index]['mask'],mode=ImageReadMode.GRAY)
        img = img.float()
        img = self.imageTransform(img)
        img = img / 255.
        
        mask[mask <= 120] = 0
        mask[mask >= 240] = 1
        
        mask = self.maskTransform(mask)
        
        assert mask.max() <= 1.0 and mask.min() >= 0
        mask = mask.float()
        
        return img,mask
    

class ForestDataModule(pl.LightningDataModule):
    def __init__(self,data_dir="metadata.csv",batch_size=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size


    def setup(self, stage: str) -> None:

        if stage == "fit":
            unet_train_full = GenerateData()
            train_set_size = int(len(unet_train_full) * 0.9)
            val_set_size = len(unet_train_full) - train_set_size
            self.train,self.validate = random_split(unet_train_full,[train_set_size,val_set_size])
        
        #if stage == "validate":

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        #ds = GenerateData()
        return DataLoader(self.train,self.batch_size,num_workers=11)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.validate,self.batch_size)

        

if __name__ == "__main__":
    df = pd.read_csv("meta_data.csv")

    ds = GenerateData()

    dl = DataLoader(ds,batch_size=1,shuffle=True)

    img, mask = next(iter(dl))
    mask = mask.squeeze(dim=0)
    print(mask.shape)
    mask = mask.to(torch.uint8)

    write_jpeg(mask,"after_preprocess.jpeg")
#print(tor)

#     mask_out = torch.sigmoid(mask)

    

#     print(mask_out[mask_out <= 0.5])
#     #print(torch.sigmoid(mask))

#     #print(img.shape)

#     print(mask)
#     print(img.shape)


# #print(df.head())

#     img_path = "Forest_Segmented/masks/" + df.loc[0]['mask']

#     img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

#     print("opencv image",img.shape)
#     resize_img = cv2.resize(img,(388,388),interpolation=cv2.INTER_CUBIC)
#     print(mask[mask == 8])
#     #print(torch.min(mask))

#     cv2.imshow("MaskImage",img)
#     cv2.imshow("Resize Image",resize_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# # cv2.imshow("Mask Image", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
