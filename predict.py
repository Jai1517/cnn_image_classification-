import sys, torch
from PIL import Image
from torchvision import transforms
from model import SmallCNN
classes=("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
if len(sys.argv)<2: raise SystemExit("Usage: python src/predict.py image.png")
model=SmallCNN(); model.load_state_dict(torch.load("artifacts/cifar10_cnn.pt",map_location="cpu",weights_only=True)); model.eval()
tf=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))])
with torch.no_grad():
    probs=model(tf(Image.open(sys.argv[1]).convert("RGB")).unsqueeze(0)).softmax(1)[0]
idx=int(probs.argmax()); print({"class":classes[idx],"confidence":round(float(probs[idx]),3)})
