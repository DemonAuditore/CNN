import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import sys

# Hyper Parameters
EPOCH = 1       # training times
BATCH_SIZE = 50
LR = 0.005      # learning rate
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),        # (0, 1) (0, 255)
    download=DOWNLOAD_MNIST
)

# plot one example
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

print(type(test_data))
sys.exit(0)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
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
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

if __name__ == "__main__":
    cnn = CNN()
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(train_loader):     # gives batch data, normalize x when item
            b_x = Variable(x)   # batch x
            b_y = Variable(y)   # batch y


            output = cnn(b_x)               # cnn output
            # print("ok!")
            # sys.exit(0)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    # print 10 predictions from test data
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
