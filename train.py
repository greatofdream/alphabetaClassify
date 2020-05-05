import numpy as np, h5py, time, argparse, sys
import torch
from torch import nn, optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import scipy.stats

batchSize= 256
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 17)# 6@984 14
        self.conv2 = nn.Conv2d(6,16,4)# stride=1, padding=3) 16@ 120 11
        self.fc1 = nn.Linear(16*15*11, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 10)
        self.fc4 = nn.Linear(10, 1)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 8))# 6@123 14
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 8))# 16@ 15 11
        x = x.view(-1, self.num_flat_features(x))# 16*30*11
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
batchSize = 256

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt', help='output')
    psr.add_argument('ipt', nargs='+', help='input')
    args = psr.parse_args()
    outFile = args.opt
    trainFiles = args.ipt
    
    model = MLP()
    print("model parameters number:{}".format(np.sum(param.numel() for param in model.parameters())))
    print(model)

    dataOrigin = np.zeros((0,30,1000))
    labelOrigin = np.zeros((0))
    for trainFile in trainFiles:
        with h5py.File(trainFile) as ipt:
            dataOrigin = np.concatenate((dataOrigin,ipt['PETruth'][:]))
            labelOrigin = np.concatenate((labelOrigin, ipt['ParticleType'][:]))
    dataOrigin = dataOrigin.reshape(-1, 1, 30, 1000)
    print('training data shape:{}'.format(dataOrigin.shape))
    print(time.asctime(time.localtime(time.time())))
    trainData, testData, trainLabel, testLabel = train_test_split(dataOrigin, labelOrigin, test_size=0.05, random_state=42)
    print('trainData,shape:{};trainLabel.shape:{}'.format(trainData.shape,trainLabel.shape))
    train_dataset = TensorDataset(torch.from_numpy(trainData).float(), torch.from_numpy(trainLabel).float())
    train_dataset = TensorDataset(torch.from_numpy(trainData).float(), torch.from_numpy(trainLabel).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True )
    test_dataset = TensorDataset(torch.from_numpy(testData).float(), torch.from_numpy(testLabel).float())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_loss = 10000.
    runningLoss_history = np.zeros(50)
    validLoss_history = np.zeros(50)
    checking_period = 300
    start_time = time.time()
    for epoch in range(15):
        # model.train()
        validnanNum = 0
        running_loss = 0.0
        for i, (data, label) in enumerate(train_loader):
            data = data.float()
            out = model(data)
            loss = criterion(out, label)
            running_loss += loss.data.item() * label.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % checking_period == 0:    # print every 2000 mini-batches
                print('[%d, %5d] running_loss: %.3f;loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (checking_period*i), loss.data.item()))
                
        # model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        lossvaild = 0.0
        with torch.no_grad():
            for (data, label) in test_loader:
                data = data.float()
                out = model(data)
                loss = criterion(out, label)
                if np.isnan(loss.item()):
                    validnanNum += 1
                    print('error valid', out)
                else:
                    valid_loss += loss.data.item() * label.size(0)
                    # lossvaild += sum([scipy.stats.wasserstein_distance(np.arange(column),np.arange(column),o.detach().numpy(), l.view((column,)).detach().numpy()) for o, l in zip(out, label)])
        runningLoss_history[epoch] = running_loss/(len(train_dataset))
        validLoss_history[epoch] = valid_loss/(len(test_dataset) - validnanNum)
        print('Finish {} epoch, Train Loss:{:.6f}, Valid Loss:{:.6f};lossvaild:{:.6f}'.format(epoch+1, runningLoss_history[epoch], validLoss_history[epoch], lossvaild/(len(test_dataset) - validnanNum)))
        cur_loss = validLoss_history[epoch]
        if cur_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = cur_loss
    end_time = time.time()
    delta_time = end_time - start_time
    print('cost time:{}s'.format(delta_time))
    torch.save(best_model.state_dict(), outFile)