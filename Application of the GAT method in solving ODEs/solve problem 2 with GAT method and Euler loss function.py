# use GAT method to solve dx/dt=-2x+4y and dy/dt=-x+3y, with initial condition x(0)=5 and y(0)=2
# the analytical solution is x=exp(2t)+4exp(-t) and y=exp(2t)+exp(-t)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)    # reproducible

t = torch.linspace(0, 1, 101).unsqueeze(dim=1)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)[:, 0]


net1 = Net(1, 100, 1)
net2 = Net(1, 100, 1)


def f(xx, yy):
    return [-2*xx+4*yy, -xx+3*yy]


# Euler loss is defined as sum((dx/dt+2x-4y)**2+(dy/dt+x-3y)**2)
def loss_func(input1, input2):
    derivative1 = [(input1[ii+1] - input1[ii]) / 0.01 for ii in range(100)]
    derivative2 = [(input2[ii+1] - input2[ii]) / 0.01 for ii in range(100)]
    k = f(input1, input2)
    return sum([(derivative1[ii]-k[0][ii])**2+(derivative2[ii]-k[1][ii])**2 for ii in range(100)]) / 100


learning_rate = [1e-4, 1e-4]
tolerance = [1e-3, 1e-3]
epoch = 0
plt.ion()
for iteration in range(2000):
    print(iteration, epoch)
    optimizer = torch.optim.Adam([{"params": net1.parameters()}, {"params": net2.parameters()}], lr=learning_rate[min(1, iteration // 4)])
    train_loss_before = 1e5
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[min(1, iteration // 4)]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction1 = net1(t)
        train_prediction2 = net2(t)
        train_loss = loss_func(train_prediction1, train_prediction2)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(t.numpy()[:, 0], (torch.exp(2 * t) + 4 * torch.exp(-t)).numpy()[:, 0])
        plt.plot(t.numpy()[:, 0], train_prediction1.data.numpy(), '.k')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(t.numpy()[:, 0], (torch.exp(2 * t) + torch.exp(-t)).numpy()[:, 0])
        plt.plot(t.numpy()[:, 0], train_prediction2.data.numpy(), '.k')
        plt.text(0, 5, 'loss=%.3g' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.grid(True)
        plt.pause(0.1)

    # reset network parameters to conform to initial condition and jump out of local optimum
    prediction1 = net1(t).data.numpy()
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = [5]                         # initial condition
    a = [(prediction1[ii + 1] - prediction1[ii]) / 0.01 for ii in range(100)]
    weight1.append([abs(a[0])])
    bias1.append(0)
    weight2[0].append(1 if a[0] >= 0 else -1)

    for ii in range(1, 100):
        weight1.append([abs(a[ii] - a[ii - 1])])
        bias1.append(-abs(a[ii] - a[ii - 1]) * ii / 100)
        weight2[0].append(1 if a[ii] - a[ii - 1] >= 0 else -1)

    net1.hidden.weight.data = torch.Tensor(weight1)
    net1.hidden.bias.data = torch.Tensor(bias1)
    net1.predict.weight.data = torch.Tensor(weight2)
    net1.predict.bias.data = torch.Tensor(bias2)

    prediction2 = net2(t).data.numpy()
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = [2]                         # initial condition
    a = [(prediction2[ii + 1] - prediction2[ii]) / 0.01 for ii in range(100)]
    weight1.append([abs(a[0])])
    bias1.append(0)
    weight2[0].append(1 if a[0] >= 0 else -1)

    for ii in range(1, 100):
        weight1.append([abs(a[ii] - a[ii - 1])])
        bias1.append(-abs(a[ii] - a[ii - 1]) * ii / 100)
        weight2[0].append(1 if a[ii] - a[ii - 1] >= 0 else -1)

    net2.hidden.weight.data = torch.Tensor(weight1)
    net2.hidden.bias.data = torch.Tensor(bias1)
    net2.predict.weight.data = torch.Tensor(weight2)
    net2.predict.bias.data = torch.Tensor(bias2)

    train_prediction1 = net1(t)
    train_prediction2 = net2(t)
    train_loss = loss_func(train_prediction1, train_prediction2)
    print(train_loss.data.numpy())
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(t.numpy()[:, 0], (torch.exp(2 * t) + 4 * torch.exp(-t)).numpy()[:, 0])
    plt.plot(t.numpy()[:, 0], train_prediction1.data.numpy(), '.k')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t.numpy()[:, 0], (torch.exp(2 * t) + torch.exp(-t)).numpy()[:, 0])
    plt.plot(t.numpy()[:, 0], train_prediction2.data.numpy(), '.k')
    plt.text(0, 5, 'loss=%.3g' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
    plt.grid(True)
    plt.pause(0.1)

plt.ioff()
plt.show()
