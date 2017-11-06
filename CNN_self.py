import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

import numpy as np
from torch.utils.data import TensorDataset
import random
import time

# Hyper Parameters
# EPOCH = 1       # training times
# BATCH_SIZE = 50
# LR = 0.001      # learning rate


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),        # (0, 1) (0, 255)
    download=False
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self, EPOCH, BATCH_SIZE, LR, fig_wid, fig_len):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # convolution layer (m*n*p)
            nn.Conv2d(              # (1, 28, 28)
                in_channels=1,      # input number of filters
                out_channels=16,    # output number of filters
                kernel_size=5,      # size of filters
                stride=1,           # gap size
                padding=2           # if strid = 1, padding = (kernel_size-1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential( # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # -> (32, 14, 14)
            nn.ReLU(),                  # -> (32, 14, 14)
            nn.MaxPool2d(2)             # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * int(fig_wid/4) * int(fig_len/4), 1)
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

    def data_process(self, train_dataset):
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        return train_loader

    def train(self, train_dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all cnn parameters
        loss_func = nn.MSELoss()
        train_loader = self.data_process(train_dataset)
        for epoch in range(self.EPOCH):
            for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when item
                b_x = Variable(x).cuda()  # batch x
                b_y = Variable(y).cuda()  # batch y

                output = self.forward(b_x)  # cnn output

                loss = loss_func(output, torch.unsqueeze(b_y.type(torch.FloatTensor).cuda(), dim=1))  # mean squared error loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                # if step % 50 == 0:
                #     pre_output = self.predict(test_x)
                #     print(pre_output)
                #     # pred_y = torch.max(test_output, 1)[1].data.squeeze()
                #     # diff_tensor = torch.unsqueeze(pred_y-test_y, dim=1)
                #     # test_loss = torch.sum(torch.squeeze(torch.mm(diff_tensor, torch.t(diff_tensor)))) / test_y.size(0)
                #     # test_loss = loss_func(pre_output, torch.unsqueeze(Variable(test_y).type(torch.FloatTensor), dim=1))
                #     # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test loss: %.4f' % test_loss.data[0])

    def predict(self, test_data):
        test_output = self.forward(test_data).data.numpy()[0]
        return test_output


# ----------------------------------------------------------------------------------------------------------------
# ARTIFICIAL DATA
# ----------------------------------------------------------------------------------------------------------------
SAMPLE_NUMBER = 10
fig_len = 196
fig_wid = 265
space_feature_list_all = [np.array([random.sample([0, 1], 1)[0] for x in range(fig_wid * fig_len)]) for x in range(SAMPLE_NUMBER)]
space_value_list = [random.random() for x in range(SAMPLE_NUMBER)]
img_shape = (fig_wid, fig_len)
space_feature_array = np.array([[x.reshape(*img_shape)] for x in space_feature_list_all])
space_value_array = np.array(space_value_list)
space_feature_tensor = torch.from_numpy(space_feature_array).float()
# space_feature_tensor = torch.from_numpy(space_feature_array)
space_value_tensor = torch.from_numpy(space_value_array)
print ("space_feature_tensor: ", space_feature_tensor.size())
print ("space_value_tensor: ", space_value_tensor.size())
space_action_dataset = TensorDataset(space_feature_tensor, space_value_tensor)
# print ("space_action_dataset: ", space_action_dataset)

# print ("train_data.data_tensor: ", train_data.data_tensor)
# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    t1 = time.time()
    cnn = CNN(1, 50, 0.001, fig_wid, fig_len)
    cnn.cuda()
    cnn.train(space_action_dataset)
    t2 = time.time()
    t = t2-t1
    print('Running Time of CNN training: ', t)
    # cnn.train(train_data)
    # prediction = cnn.predict(test_x)
    # print(prediction)
