import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy.ma.core import shape

# 数据生成及分类
x1 = torch.rand(10000,1)
x2 = torch.rand(10000,1)
x3 = torch.rand(10000,1)
y1 = ((x1+x2+x3) < 1).float()
y2 = ((1<(x1+x2+x3)) & ((x1+x2+x3) < 2)).float()
y3 = ((x1+x2+x3)>2).float()
data = torch.cat([x1,x2,x3,y1,y2,y3],axis=1)

train_size = int(len(data)*0.7)
test_size = len(data)-train_size
data = data[torch.randperm(data.size(0)),:]
train_data = data[0:train_size, :]
test_data = data[train_size:, :]
print(train_data, test_data, shape(train_data), shape(test_data), sep='\n')


# 模型设计

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential( # 全连接层
            nn.Linear(3, 5), nn.ReLU(),
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 5), nn.ReLU(),
            nn.Linear(5, 3)
        )
    def forward(self, x):
        y = self.net(x)
        return y

model = DNN()
loss_fn = nn.MSELoss() # 损失函数选择
learning_rate = 0.001 # 学习率选择
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 模型训练
epochs = 1000
losses = []
x = train_data[:, :3] # randn生成的原始数据
y = train_data[:, -3:] # 形成的独热编码

for epoch in range(epochs): # 网络训练开始
    pred = model(x)
    loss = loss_fn(pred, y) # 计算
    losses.append(loss.item()) # 将偏差转化为普通的python类型
    optimizer.zero_grad()   # 梯度清零
    loss.backward() # 反向传播
    optimizer.step() # 参数优化

fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# 测试网络
with torch.no_grad(): # no_grad means 不开启梯度计算
    pred = model(x)
    pred[:, torch.argmax(pred, axis=1)] = 1
    pred[pred!=1] = 0
    correct = torch.sum((pred == y).all(1))
    total = y.size(0)
    print(f'测试集精准度:{100*correct/total} %')