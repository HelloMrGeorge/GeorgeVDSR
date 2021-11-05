import os
import torch
from random import randint
from torchvision import transforms
from PIL import Image

source = 'source' #source是存放图片的文件夹
paths = os.listdir(source)
tsTen = transforms.ToTensor()
data_ls = []
target_ls = []

for path in paths:
    path = os.path.join(source, path)
    im = Image.open(path)
    # im = im.convert(mode='YCbCr') #如需转换成YCbCr空间，则去掉注释
    size = im.size
    im_scaled = im.resize((int(size[0]/2), int(size[1]/2))).resize(size)
    idx = randint(0,2)
    im_mat = tsTen(im)[idx]
    ims_mat = tsTen(im_scaled)[idx]
    pix = 32 #设置训练集的每张图片的尺寸
    for i in range(size[0]//pix):
        for j in range(size[1]//pix):
            data_mat = im_mat[i*pix:(i+1)*pix, j*pix:(j+1)*pix]
            data_mat = data_mat.reshape(1, *data_mat.shape)
            data_ls.append(data_mat)
            target_mat = im_mat[i*pix:(i+1)*pix, j*pix:(j+1)*pix] - ims_mat[i*pix:(i+1)*pix, j*pix:(j+1)*pix]
            target_mat = target_mat.reshape(1, *target_mat.shape)
            target_ls.append(target_mat)

torch.save(data_ls, 'source/data.pls')
torch.save(target_ls, 'source/target.pls')
    