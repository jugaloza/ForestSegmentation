from models import UNetModel
import argparse
import torch

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--CKPT_PATH","-p",type=str,help="Specify path of checkpoint to which convert it into PTH file")

    args = vars(ap.parse_args())

    model = UNetModel.load_from_checkpoint(args["CKPT_PATH"])

    #child = model.children()[0]
    
    for child in model.children():
        #print(child)
        torch.save(child.state_dict(),"model.pth")
        break
    