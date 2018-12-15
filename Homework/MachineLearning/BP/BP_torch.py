import torch 
import torch.utils.data as Data
import pandas as pd
import numpy as np
from torch.autograd import Variable

EPOCH = 1000
BATCH_SIZE = 300
LR = 0.5

def preProcess(filepath):
    Data = pd.read_csv(filepath, header=None, sep='\s+', dtype=np.object)
    Data = Data.replace('?', '-1')
    Data = Data[Data[22] != '-1']
    Data = Data.reset_index(drop=True)
    Data = pd.DataFrame(Data, dtype=np.float)
    Data[2] /= 1000000
    Data[3] /= 100
    Data[4] /= 100
    Data[5] /= 100
    Data[18] /= 100
    Data[19] /= 100
    Data[24] /= 10000
    return Data

if __name__ == '__main__':
    net = torch.nn.Sequential(
        torch.nn.Linear(27, 4),
        torch.nn.Sigmoid(),
        torch.nn.Linear(4, 3),
        torch.nn.Sigmoid()
    )
    net.cuda()
    train_sets = preProcess('horse-colic.data')
    test_sets = preProcess('horse-colic.test')

    train_length = len(train_sets)
    test_length = len(test_sets)

    train_x = torch.FloatTensor(np.array(train_sets.drop([22], axis=1)))
    train_y = torch.LongTensor(np.array(train_sets[22])) - 1
    test_x = torch.FloatTensor(np.array(test_sets.drop([22], axis=1))).cuda()
    test_y = (torch.LongTensor(np.array(test_sets[22])) - 1).cuda()

    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            batch_x = Variable(x).cuda()
            batch_y = Variable(y).cuda()
            # print(step)
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                test_output = net(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data
                # print(pred_y == test_y.numpy())
                accuracy = (pred_y == test_y).type(torch.FloatTensor).sum() / test_y.size(0)
                print('Epoch', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = net(test_x)
    pred_y = torch.max(test_output, 1)[1].cuda().data
    print(pred_y)