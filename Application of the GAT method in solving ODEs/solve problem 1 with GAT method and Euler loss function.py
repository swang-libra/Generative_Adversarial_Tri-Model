# use GAT method to solve y'=-xy/(x+2), with initial condition y(0)=4
# the analytical solution is y=exp(-x)(x+2)^2

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)    # reproducible

x = torch.linspace(0, 1, 101).unsqueeze(dim=1)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)[:, 0]


net = Net(1, 100, 1)


def f(xx, yy):
    return -xx*yy/(xx+2)


# Euler loss is defined as sum((y'+xy/(x+2))**2)
def loss_func(input):
    derivative = [(input[ii+1] - input[ii]) / 0.01 for ii in range(100)]
    k = f(x[:, 0], input)
    return sum([(derivative[ii] - k[ii]) ** 2 for ii in range(100)]) / 100


learning_rate = [1e-3, 1e-4]
tolerance = [1e-3, 1e-5]
epoch = 0
plt.ion()
for iteration in range(9):
    print(iteration, epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate[min(iteration // 4, 1)])
    train_loss_before = 1e5
    train_loss = train_loss_before - 1
    while abs(train_loss_before - train_loss) > tolerance[min(iteration // 4, 1)]:
        epoch += 1
        train_loss_before = train_loss
        train_prediction = net(x)
        train_loss = loss_func(train_prediction)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        plt.cla()
        plt.plot(x.numpy()[:, 0], (torch.exp(-x) * (x + 2) ** 2).numpy()[:, 0])
        plt.plot(x.numpy()[:, 0], train_prediction.data.numpy(), '.k')
        plt.text(0, 3.3, 'loss=%.3g' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.grid(True)
        plt.pause(0.1)

    # reset network parameters to conform to initial condition and jump out of local optimum
    prediction = net(x).data.numpy()
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = [4]         # initial condition
    a = [(prediction[ii+1] - prediction[ii]) / 0.01 for ii in range(100)]
    weight1.append([abs(a[0])])
    bias1.append(0)
    weight2[0].append(1 if a[0] >= 0 else -1)

    for ii in range(1, 100):
        weight1.append([abs(a[ii] - a[ii - 1])])
        bias1.append(-abs(a[ii] - a[ii-1]) * ii / 100)
        weight2[0].append(1 if a[ii] - a[ii - 1] >= 0 else -1)

    net.hidden.weight.data = torch.Tensor(weight1)
    net.hidden.bias.data = torch.Tensor(bias1)
    net.predict.weight.data = torch.Tensor(weight2)
    net.predict.bias.data = torch.Tensor(bias2)

    train_prediction = net(x)
    train_loss = loss_func(train_prediction)
    print(train_loss.data.numpy())
    plt.cla()
    plt.plot(x.numpy()[:, 0], (torch.exp(-x) * (x + 2) ** 2).numpy()[:, 0])
    plt.plot(x.numpy()[:, 0], train_prediction.data.numpy(), '.k')
    plt.text(0, 3.3, 'loss=%.3g' % train_loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
    plt.grid(True)
    plt.pause(0.1)

plt.ioff()
plt.show()