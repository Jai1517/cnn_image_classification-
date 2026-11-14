import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SmallCNN
CLASSES=("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--batch-size",type=int,default=64); args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))])
    train=datasets.CIFAR10("data",train=True,download=True,transform=tf)
    test=datasets.CIFAR10("data",train=False,download=True,transform=tf)
    tr=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0)
    te=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=SmallCNN().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); total=0
        for x,y in tr:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); correct=count=0
        with torch.no_grad():
            for x,y in te:
                pred=model(x.to(device)).argmax(1).cpu(); correct+=(pred==y).sum().item(); count+=len(y)
        print(f"epoch={epoch+1} train_loss={total/len(tr):.4f} test_accuracy={correct/count:.3f}")
    out=Path("artifacts/cifar10_cnn.pt"); out.parent.mkdir(exist_ok=True); torch.save(model.state_dict(),out); print("Saved",out)
if __name__=="__main__": main()
