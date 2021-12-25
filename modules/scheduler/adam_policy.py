from torch.optim.lr_scheduler import _LRScheduler

class MyLR(_LRScheduler):
    def __init__(self, optimizer, cycle, current_epoch=0, last_epoch=-1):
        self.current_epoch = current_epoch
        self.cycle = cycle
        super(MyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = int(self.current_epoch / 2)
        if current_epoch == 0:
            c_lr = 0.0004
        if current_epoch == 1:
            c_lr = 0.0008
        if current_epoch == 2:
            c_lr = 0.0012
        if current_epoch == 3:
            c_lr = 0.0016
        if current_epoch == 4:
            c_lr = 0.002
        if current_epoch == 5:
            c_lr = 0.0016
        if current_epoch == 6:
            c_lr = 0.0012
        if current_epoch == 7:
            c_lr = 0.0008
        if current_epoch == 8:
            c_lr = 0.0012
        if current_epoch == 9:
            c_lr = 0.0006
        if current_epoch == 10:
            c_lr = 0.0008
        if current_epoch == 11:
            c_lr = 0.0004
        if current_epoch == 12:
            c_lr = 0.0002
        if current_epoch == 13:
            c_lr = 0.0004
        if current_epoch == 14:
            c_lr = 0.0002
        if current_epoch >= 15:
            c_lr = 0.0002
        return [c_lr for base_lr in self.base_lrs]

    # 当需要传入其他参数时，需要重新定义step 函数，如ReduceLROnPlateau方法
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        if epoch == self.cycle:
            epoch = 0
        self.current_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import LambdaLR
# import itertools
# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
# 
#     def forward(self, x):
#         pass
# 
# net_1 = model()
# optimizer = torch.optim.AdamW(net_1.parameters(), lr=0.0001, weight_decay=0.005)
# scheduler = MyLR(optimizer, 30)
# print("初始化的学习率：", optimizer.defaults['lr'])
# 
# ep =[]
# lr =[]
# for epoch in range(0, 60):
#     # train
#     optimizer.zero_grad()
#     optimizer.step()
# 
#     print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
#     ep.append(epoch)
#     lr.append(optimizer.param_groups[0]['lr'])
#     scheduler.step()
# 
# import matplotlib.pyplot as plt
# plt.plot(ep,lr)
# plt.show()