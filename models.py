from torch import nn 
from torch.nn import functional as F
from torchvision import transforms
import torch
import lightning.pytorch as pl
from Loss import DiceLoss
from torch.nn.functional import binary_cross_entropy
from torchmetrics.classification import BinaryJaccardIndex



class UNet(nn.Module):
    def __init__(self):
        super().__init__()                 # output shape

        #self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3,32,3,1,1) # B,32,128,128 
        self.conv2 = nn.Conv2d(32,32,3,1,1) # B,32,64,64 
        
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3,1,1) # B, 64, 64,64
        self.conv4 = nn.Conv2d(64,64,3,1,1) # B, 64, 32, 32
        

        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,128,3,1,1) # B, 128, 32, 32
        self.conv6 = nn.Conv2d(128,128,3,1,1) # B, 128, 16, 16
        

        self.conv7 = nn.Conv2d(128,256,3,1,1) # B, 256, 16, 16
        self.conv8 = nn.Conv2d(256,256,3,1,1) # B, 256, 8, 8
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(256,512,3,1,1) # B, 512, 8, 8
        self.conv10 = nn.Conv2d(512,512,3,1,1) # B, 512, 8, 8
        self.batchnorm5 = nn.BatchNorm2d(256)


        self.upconv1 = nn.ConvTranspose2d(512,256,2,2) # B, 256, 16, 16
        self.batchnorm6 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512,256,3,1,1)
        self.conv12 = nn.Conv2d(256,256,3,1,1)
        self.batchnorm7 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(256,128,2,2)
        self.batchnorm8 = nn.BatchNorm2d(256)

        self.conv13 = nn.Conv2d(256,128,3,1,1)
        self.conv14 = nn.Conv2d(128,128,3,1,1)
        self.batchnorm9 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(128,64,2,2)
        self.batchnorm10 = nn.BatchNorm2d(128)

        self.conv15 = nn.Conv2d(128,64,3,1,1)
        self.conv16 = nn.Conv2d(64,64,3,1,1)
        self.batchnorm11 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(64,32,2,2)
        self.batchnorm12 = nn.BatchNorm2d(64)

        self.conv17 = nn.Conv2d(64,32,3,1,1)
        self.conv18 = nn.Conv2d(32,32,3,1,1)
        self.batchnorm13 = nn.BatchNorm2d(64)

        self.conv19 = nn.Conv2d(32,1,3,1,1)

        #self.conv11 = nn.Conv2d(256,)

    def forward(self,x):

        #x = self.batchnorm1(x)
        #x = F.dropout2d(F.leaky_relu(self.conv1(x)),0.2) 
        x = F.leaky_relu(self.conv1(x)) # B, 32, 128, 128
        x = F.leaky_relu(self.conv2(x)) # B, 32, 128, 128
        x_skip_3 = x

        x = F.max_pool2d(x,(2,2)) # B,32,64,64
        
        x = self.batchnorm2(x)
        x = F.leaky_relu(self.conv3(x)) # B,64, 64, 64
        x = F.leaky_relu(self.conv4(x)) # B, 64, 64, 64
        x_skip_2 = x

        x = F.max_pool2d(x,(2,2)) # B, 64, 32, 32
        
        x = self.batchnorm3(x)
        x = F.leaky_relu(self.conv5(x)) # B, 128, 32, 32
        #x = F.dropout2d(F.leaky_relu(self.conv5(x)),0.2) 
        x = F.leaky_relu(self.conv6(x)) # B, 128, 32, 32
        x_skip_1 = x 

        x = F.max_pool2d(x,(2,2)) # B, 128, 16, 16
        
        x = self.batchnorm4(x)
        x = F.leaky_relu(self.conv7(x)) # B, 256, 16, 16  
        x = F.leaky_relu(self.conv8(x))  # B, 256, 16, 16 
        x_skip = x

        x = F.max_pool2d(x,(2,2)) # B, 256, 8, 8
        
        x = self.batchnorm5(x)
        x = F.leaky_relu(self.conv9(x)) # B, 512, 8, 8
        x = F.leaky_relu(self.conv10(x)) # B, 512, 8, 8
        
        x = self.batchnorm6(x)
        x_up = F.leaky_relu(self.upconv1(x))

        x = torch.cat([x_up,x_skip],dim=1)

        x = self.batchnorm7(x)
        x = F.leaky_relu(self.conv11(x))
        x = F.leaky_relu(self.conv12(x))

        x = self.batchnorm8(x)
        x_up_1 = F.leaky_relu(self.upconv2(x))

        x = torch.cat([x_up_1,x_skip_1],dim=1)


        x = self.batchnorm9(x)
        x = F.leaky_relu(self.conv13(x))
        x = F.leaky_relu(self.conv14(x))

        x = self.batchnorm10(x)
        x_up_2 = F.leaky_relu(self.upconv3(x))

        x = torch.cat([x_up_2,x_skip_2],dim=1)

        x = self.batchnorm11(x)
        x = F.leaky_relu(self.conv15(x))
        x = F.leaky_relu(self.conv16(x))

        x = self.batchnorm12(x)
        x_up_3 = F.leaky_relu(self.upconv4(x))
        
        x = torch.cat([x_up_3,x_skip_3],dim=1)

        x = self.batchnorm13(x)
        x = F.leaky_relu(self.conv17(x))
        x = F.leaky_relu(self.conv18(x))

        x = F.sigmoid(self.conv19(x))

        return x
    

class UNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        self.metrics = BinaryJaccardIndex()
        


    def training_step(self, batch, batchIdx):
        x,y = batch
        y_pred = self.unet(x)

        jaccardIndex = self.metrics(y_pred,y)
        loss = binary_cross_entropy(y_pred,y)

        self.log("Training Loss", loss)
        self.log("train_jaccard_index",jaccardIndex)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(),0.01)
        
        return optimizer
    
    def validation_step(self, batch, batchIdx):
        x,y = batch
        y_pred = self.unet(x)
        
        jaccardIndex = self.metrics(y_pred,y)
        val_loss = binary_cross_entropy(y_pred,y)
        
        self.log("val_loss",val_loss)
        self.log("val_jaccard_index",jaccardIndex)
        
                


if __name__ == "__main__":

    model = UNet()

    x = torch.randn((1,3,128,128))

    print(model(x).shape)