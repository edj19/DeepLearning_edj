import numpy as np
import matplotlib.pylab as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.042],
                    [10.791],[5.313],[7.997],[3.1]],dtype=np.float32)

y_train = np.array([[1.7],[2.67],[2.09],[3.19],[1.694],[1.573],
                     [3.366],[2.596],[2.53],[1.221],[2.827],
                     [3.465],[1.65],[2.904],[1.3]],dtype=np.float32)
#plt.scatter(x_train,y_train,c='green',alpha=0.75)
#plt.show()

# 先将numpy.array 转换为Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

#定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)  # input and output is 1 dimension
        
    def forward(self,x):
        out = self.linear(x)
        return out
if torch.cuda.is_available():
    print("True")
    model = LinearRegression().cuda()
else:
    print("False")
    model = LinearRegression()

#定义损失函数和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)


# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out,target)

    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%20 ==0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1,num_epochs,loss.item()))



#测试结果
model.eval()   #将模型变为测试模型，这是因为有一些层操作，比如Dropout和BatchNormalization在训练和测试的时候是不一样的，所以需要我们通过这样一个操作来转换这些不一样的层操作
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(),y_train.numpy(),'ro',label='Original data')
plt.plot(x_train.numpy(),predict,label='Fitting Line')
plt.title(u'一元回归')
plt.grid(True)
plt.show()