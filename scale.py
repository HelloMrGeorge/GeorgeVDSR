import torch

from PIL import Image
from torchvision import transforms

def crop(n):
    #crop去除不在0-1之间的灰度
    if n > 1:
        n = 1
    elif n < 0:
        n = 0
    return n

class VDSRscaler:

    def __init__(self) -> None:
        self.tsTen = transforms.ToTensor()
        self.tsImg = transforms.ToPILImage()
        self.model = torch.load('vdsrcnn.pkl', map_location=torch.device('cpu'))

    def scale(self, path, rate=2):
        #path指定源图像路径，rate指定放大倍率
        im = Image.open(path)
        size = im.size
        size = (int(size[0]*rate), int(size[1]*rate))
        im = im.resize(size)

        ilr = self.tsTen(im)
        X = torch.zeros(*ilr.shape)
        for i in range(3):
            X_Y = self.model(ilr[i].reshape(1, 1, *ilr[i].shape))
            X[i] = X_Y.reshape(*X[i].shape)

        res = self.tsImg(X)
        res.save(f'{path[:-4]}_res.png')
        del res

        X = ilr + X
        del ilr
        sample = self.tsImg(X.apply_(crop))
        sample.save(f'{path[:-4]}_new.png')

if __name__ == '__main__':
    with torch.no_grad():
        scaler = VDSRscaler()
        scaler.scale('sample/77845097_p0.jpg')
    
