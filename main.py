from torch import nn
import torch
from myVDSR import VDSR
from dataset import TrainDataset
from torch.utils.data import DataLoader


batch_size = 128
lr = 0.1 #学习率每10轮乘以0.1
epochs = 5 #10+10+10+10 10+10+10
momentum = 0.9
weight_decay = 1e-4
layers = 16
clip = 0.4

model = VDSR(layers).cuda(0)
# model = torch.load('vdsrcnn.pkl').cuda(0)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
training_dataset = TrainDataset('source/data.pls', 'source/target.pls')
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

def train(dataloader, model, criterion, optimizer, clip):
    size = len(dataloader.dataset)
    loss_sum = 0
    for iteration, batch in enumerate(dataloader):
        # Compute prediction and loss
        input, target = batch[0], batch[1]
        target.requires_grad_(False)
        pred = model(input)
        loss = criterion(pred, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),clip) #梯度修剪
        optimizer.step()

        loss_sum = loss_sum + loss.item() #计算总损失
        if iteration % 10 == 0:
            loss, current = loss.item(), iteration * batch_size
            print(f"loss: {loss:>14f}  [{current:>5d}/{size:>5d}]")
    print(f'loss_sum: {loss_sum:>14f}')

if __name__ == '__main__':
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_loader, model, criterion, optimizer, clip)
        
    torch.save(model,'vdsrcnn.pkl')
    print("Done!")