import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2

# video analysis
pressure_brightness = scio.loadmat('./pressure_brightness_mapping.mat')['pressure'].squeeze()

video = cv2.VideoCapture('./balance_sensor_video_data/swing_front_back_calibrated.mp4')
# video = cv2.VideoCapture('./balance_sensor_video_data/swing_left_right_calibrated.mp4')
# video = cv2.VideoCapture('./balance_sensor_video_data/normal_state_calibrated.mp4')
# video = cv2.VideoCapture('./balance_sensor_video_data/drunk_state_calibrated.mp4')
center_COP_x = []
center_COP_y = []
current_frame = 0
while current_frame < 301:
    success, frame = video.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)  # 0-based index of the frame to be decoded
    pressure_map = np.array(list(map(lambda x: pressure_brightness[x], image)))
    center_COP_x.append((pressure_map.sum(axis=0) * np.arange(598)).sum() / pressure_map.sum())
    center_COP_y.append((pressure_map.sum(axis=1) * np.arange(420)).sum() / pressure_map.sum())

center_COP_x = np.array(center_COP_x)
center_COP_x = center_COP_x - center_COP_x.mean()
center_COP_y = np.array(center_COP_y)
center_COP_y = center_COP_y - center_COP_y.mean()
# the y direction of human is inverse to the y axis of image
center_COP_y = -center_COP_y

# convert coordinate of gravity center into unit m (1 pixel corresponds to 0.697mm)

center_COP_x = center_COP_x * 6.97e-4
center_COP_y = center_COP_y * 6.97e-4

########################################################################################################################

# approximate human motion calculation
# the human model is approximately linearized to
# d(u1)/dt - u3 = 0
# d(u2)/dt - u4 = 0
# d(u3)/dt - 33g/23h*u1 = 60g/23h^2*COPy
# d(u4)/dt - 24g/11h*u2 = 60g/11h^2*COPx
# the definite conditions are that the integrals of theta1, theta2, d(theta1)/dt and d(theta2)/dt over time are 0

h = 1.78
g = 9.8
t_begin = 0
t_final = 10
dt = 1/30

N = int((t_final - t_begin) / dt)
A = np.zeros((4*N+4, 4*N+4))
b = np.zeros((4*N+4, 1))

# the implicit variables are [u1,0; u2,0; u3,0; u4,0; u1,1; u2,1; u3,1; u4,1;... u1,300; u2,300; u3,300; u4,300]
for ii in range(N):
    A[4 * ii, 4 * ii] = -1
    A[4 * ii, 4 * ii + 2] = -dt
    A[4 * ii, 4 * ii + 4] = 1
    A[4 * ii + 1, 4 * ii + 1] = -1
    A[4 * ii + 1, 4 * ii + 3] = -dt
    A[4 * ii + 1, 4 * ii + 5] = 1
    A[4 * ii + 2, 4 * ii] = -33*g/23/h*dt
    A[4 * ii + 2, 4 * ii + 2] = -1
    A[4 * ii + 2, 4 * ii + 6] = 1
    b[4 * ii + 2, 0] = 60*g/23/h**2*dt*center_COP_y[ii]
    A[4 * ii + 3, 4 * ii + 1] = -24*g/11/h*dt
    A[4 * ii + 3, 4 * ii + 3] = -1
    A[4 * ii + 3, 4 * ii + 7] = 1
    b[4 * ii + 3, 0] = 60*g/11/h**2*dt*center_COP_x[ii]
    A[4*N, 4 * ii] = 1
    A[4*N+1, 4 * ii + 1] = 1
    A[4*N+2, 4 * ii + 2] = 1
    A[4*N+3, 4 * ii + 3] = 1

u_approximate = np.linalg.solve(A, b).squeeze()
theta1 = np.array([u_approximate[4*ii] for ii in range(N+1)])
theta2 = np.array([u_approximate[4*ii+1] for ii in range(N+1)])
theta1_dot = np.array([u_approximate[4*ii+2] for ii in range(N+1)])
theta2_dot = np.array([u_approximate[4*ii+3] for ii in range(N+1)])
# the COG variation along x axis is 2/5*h*theta2
# the COG variation along y axis is 11/20*h*theta1
# the positive direction of theta1 and theta2 are inverse to y and x, so we need to add a minus sign
center_COG_x = -2/5*h*theta2
center_COG_y = -11/20*h*theta1

# plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.plot(np.linspace(t_begin, t_final, N+1), center_COP_x, 'b')
# plt.plot(np.linspace(t_begin, t_final, N+1), center_COG_x, 'r')
# plt.xlabel('time/s')
# plt.ylabel('x coordinate/m')
# plt.legend(['COP', 'COG'])
# plt.grid(True)
# plt.subplot(2, 2, 2)
# plt.plot(np.linspace(t_begin, t_final, N+1), center_COP_y, 'b')
# plt.plot(np.linspace(t_begin, t_final, N+1), center_COG_y, 'r')
# plt.xlabel('time/s')
# plt.ylabel('y coordinate/m')
# plt.legend(['COP', 'COG'])
# plt.grid(True)
# plt.subplot(2, 2, 3)
# plt.plot(np.linspace(t_begin, t_final, N+1), theta1_dot)
# plt.grid(True)
# plt.subplot(2, 2, 4)
# plt.plot(np.linspace(t_begin, t_final, N+1), theta2_dot)
# plt.grid(True)
########################################################################################################################

# full solution of human motion with GAT model
t = torch.linspace(t_begin, t_final, N+1).unsqueeze(dim=1)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.predict(x)[:, 0]


net1 = Net(1, N, 1)             # output u1
net2 = Net(1, N, 1)             # output u2
net3 = Net(1, N, 1)             # output u3
net4 = Net(1, N, 1)             # output u4
net = [net1, net2, net3, net4]

# initialize networks with approximate solution
for ii in range(4):
    weight1 = []
    bias1 = []
    weight2 = [[]]
    bias2 = [u_approximate[ii]]
    a = [(u_approximate[4*jj+ii+4] - u_approximate[4*jj+ii]) / dt for jj in range(N)]
    weight1.append([abs(a[0])])
    bias1.append(0)
    weight2[0].append(1 if a[0] >= 0 else -1)

    for jj in range(1, N):
        weight1.append([abs(a[jj] - a[jj-1])])
        bias1.append(-abs(a[jj] - a[jj-1]) * jj * dt)
        weight2[0].append(1 if a[jj] - a[jj - 1] >= 0 else -1)

    net[ii].hidden.weight.data = torch.Tensor(weight1)
    net[ii].hidden.bias.data = torch.Tensor(bias1)
    net[ii].predict.weight.data = torch.Tensor(weight2)
    net[ii].predict.bias.data = torch.Tensor(bias2)

# define 4 nets for saving the best networks during training
net1_optimum = copy.deepcopy(net1)          # initialization
net2_optimum = copy.deepcopy(net2)          # initialization
net3_optimum = copy.deepcopy(net3)          # initialization
net4_optimum = copy.deepcopy(net4)          # initialization

# plt.figure(2)
# plt.subplot(2, 2, 1)
# plt.plot(t[:, 0], net1(t).data)
# plt.grid(True)
# plt.subplot(2, 2, 2)
# plt.plot(t[:, 0], net2(t).data)
# plt.grid(True)
# plt.subplot(2, 2, 3)
# plt.plot(t[:, 0], net3(t).data)
# plt.grid(True)
# plt.subplot(2, 2, 4)
# plt.plot(t[:, 0], net4(t).data)
# plt.grid(True)


def f(u1, u2, u3, u4, COP_x, COP_y):
    return [u3, u4, 31 / 23 * u2 * u3 * u4 + 33 * g / 23 / h * u1 + 60 * g / 23 / h ** 2 * COP_y,
            - 31 / 22 * u2 * u3 ** 2 + 24 * g / 11 / h * u2 + 60 * g / 11 / h ** 2 * COP_x]


# Runge-Kutta loss function
def loss_func(input1, input2, input3, input4):
    derivative1 = [(input1[jj+1] - input1[jj]) / dt for jj in range(N)]
    derivative2 = [(input2[jj+1] - input2[jj]) / dt for jj in range(N)]
    derivative3 = [(input3[jj+1] - input3[jj]) / dt for jj in range(N)]
    derivative4 = [(input4[jj+1] - input4[jj]) / dt for jj in range(N)]
    k1 = [f(input1[jj], input2[jj], input3[jj], input4[jj], center_COP_x[jj], center_COP_y[jj]) for jj in range(N)]
    k2 = [f(input1[jj] + dt / 2 * k1[jj][0], input2[jj] + dt / 2 * k1[jj][1],
            input3[jj] + dt / 2 * k1[jj][2], input4[jj] + dt / 2 * k1[jj][3],
            center_COP_x[jj], center_COP_y[jj]) for jj in range(N)]
    k3 = [f(input1[jj] + dt / 2 * k2[jj][0], input2[jj] + dt / 2 * k2[jj][1],
            input3[jj] + dt / 2 * k2[jj][2], input4[jj] + dt / 2 * k2[jj][3],
            center_COP_x[jj], center_COP_y[jj]) for jj in range(N)]
    k4 = [f(input1[jj] + dt * k3[jj][0], input2[jj] + dt * k3[jj][1],
            input3[jj] + dt * k3[jj][2], input4[jj] + dt * k3[jj][3],
            center_COP_x[jj + 1], center_COP_y[jj + 1]) for jj in range(N)]
    difference = [[derivative1[jj] - (k1[jj][0] + 2 * k2[jj][0] + 2 * k3[jj][0] + k4[jj][0]) / 6 for jj in range(N)],
                  [derivative2[jj] - (k1[jj][1] + 2 * k2[jj][1] + 2 * k3[jj][1] + k4[jj][1]) / 6 for jj in range(N)],
                  [derivative3[jj] - (k1[jj][2] + 2 * k2[jj][2] + 2 * k3[jj][2] + k4[jj][2]) / 6 for jj in range(N)],
                  [derivative4[jj] - (k1[jj][3] + 2 * k2[jj][3] + 2 * k3[jj][3] + k4[jj][3]) / 6 for jj in range(N)]]
    return sum([difference[0][jj]**2 + difference[1][jj]**2 +
               difference[2][jj]**2 + difference[3][jj]**2
               for jj in range(N)]) / N


# define loss_value_optimum to save the best loss value
loss_value_optimum = loss_func(net1(t), net2(t), net3(t), net4(t))          # initialization
print(loss_value_optimum.data.numpy())

iteration = 0
epoch = 0
trial = 0           # the count of trials to find better solutions
optimizer = torch.optim.SGD([{"params": net1.parameters()},
                            {"params": net2.parameters()},
                            {"params": net3.parameters()},
                            {"params": net4.parameters()}], lr=1e-8)
train_loss_before = loss_value_optimum
train_loss = loss_value_optimum
plt.ion()
while True:
    print('iteration:', iteration, ', epoch:', epoch)
    while train_loss_before >= train_loss:
        epoch += 1
        train_loss_before = train_loss
        train_prediction1 = net1(t)
        train_prediction2 = net2(t)
        train_prediction3 = net3(t)
        train_prediction4 = net4(t)
        train_loss = loss_func(train_prediction1, train_prediction2, train_prediction3, train_prediction4)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # plt.clf()
        # plt.subplot(2, 2, 1)
        # plt.plot(t[:, 0], train_prediction1.data)
        # plt.grid(True)
        # plt.subplot(2, 2, 2)
        # plt.plot(t[:, 0], train_prediction2.data)
        # plt.grid(True)
        # plt.subplot(2, 2, 3)
        # plt.plot(t[:, 0], train_prediction3.data)
        # plt.grid(True)
        # plt.subplot(2, 2, 4)
        # plt.plot(t[:, 0], train_prediction4.data)
        # plt.grid(True)
        # plt.xlabel('loss=%.4e' % train_loss.data.numpy())
        # plt.pause(0.1)

    # reset network parameters to conform to 0-integral condition and jump out of local optimum
    for ii in range(4):
        prediction = net[ii](t).data.numpy()
        integral = prediction[0:-1].sum() * dt
        adjust_prediction = prediction - integral/(t_final-t_begin)

        weight1 = []
        bias1 = []
        weight2 = [[]]
        bias2 = [adjust_prediction[0]]
        a = [(adjust_prediction[jj+1] - adjust_prediction[jj]) / dt for jj in range(N)]
        weight1.append([abs(a[0])])
        bias1.append(0)
        weight2[0].append(1 if a[0] >= 0 else -1)

        for jj in range(1, N):
            weight1.append([abs(a[jj] - a[jj - 1])])
            bias1.append(-abs(a[jj] - a[jj - 1]) * jj * dt)
            weight2[0].append(1 if a[jj] - a[jj - 1] >= 0 else -1)

        net[ii].hidden.weight.data = torch.Tensor(weight1)
        net[ii].hidden.bias.data = torch.Tensor(bias1)
        net[ii].predict.weight.data = torch.Tensor(weight2)
        net[ii].predict.bias.data = torch.Tensor(bias2)

    train_loss_new = loss_func(net1(t), net2(t), net3(t), net4(t))
    print(train_loss_new.data.numpy())
    # if find better solutions, then update saved nets and solutions
    if train_loss_new < loss_value_optimum:
        net1_optimum = copy.deepcopy(net1)
        net2_optimum = copy.deepcopy(net2)
        net3_optimum = copy.deepcopy(net3)
        net4_optimum = copy.deepcopy(net4)
        theta1 = net1(t).data.numpy()
        theta2 = net2(t).data.numpy()
        theta1_dot = net3(t).data.numpy()
        theta2_dot = net4(t).data.numpy()
        loss_value_optimum = train_loss_new
        trial = 0           # reset trial to 0
    else:
        trial += 1
        if trial >= 10:     # try 10 times to find better solutions
            break           # if better solutions cannot be found in 10 consecutive times, then stop the whole process

    train_loss_before = train_loss_new
    train_loss = train_loss_new
    iteration += 1

plt.ioff()
# the COG variation along x axis is 2/5*h*theta2
# the COG variation along y axis is 11/20*h*theta1
# the positive direction of theta1 and theta2 are inverse to y and x, so we need to add a minus sign
center_COG_x = -2/5*h*theta2
center_COG_y = -11/20*h*theta1
plt.figure(3)
plt.subplot(1, 2, 1)
plt.plot(np.linspace(t_begin, t_final, N+1), center_COP_x, 'b')
plt.plot(np.linspace(t_begin, t_final, N+1), center_COG_x, 'r')
plt.xlabel('time/s')
plt.ylabel('x coordinate/m')
plt.legend(['COP', 'COG'])
# plt.axis([0, 10, -0.04, 0.04])
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(np.linspace(t_begin, t_final, N+1), center_COP_y, 'b')
plt.plot(np.linspace(t_begin, t_final, N+1), center_COG_y, 'r')
plt.xlabel('time/s')
plt.ylabel('y coordinate/m')
plt.legend(['COP', 'COG'])
# plt.axis([0, 10, -0.04, 0.04])
plt.grid(True)
plt.show()
