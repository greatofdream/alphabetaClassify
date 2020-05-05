import numpy as np, h5py, time, argparse, sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import tables

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
class AnswerData(tables.IsDescription):
    EventID=tables.Int64Col(pos=0)
    Alpha=tables.Float32Col(pos=1)
def writeSubfile(truth, eventId, subfile, mode='notbinary'):
    answerh5file = tables.open_file(subfile, mode='w', title="Answer")
    AnswerTable = answerh5file.create_table('/', 'Answer', AnswerData, 'Answer')
    answer = AnswerTable.row
    
    length = len(eventId)
    if mode == 'notbinary':
        for i in range(length):
            answer['EventID'] = eventId[i]
            answer['Alpha'] = truth[i] 
            answer.append()    
        print('\rThe writing processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    else:
        print("wait for complete")
    AnswerTable.flush()
    answerh5file.close()
def ReadPETruth(filename):
    ipt = h5py.File(filename, "r")
    EventID = np.unique(ipt['Waveform']['EventID'][:])
    petruthfull = np.zeros((EventID.shape[0],30,1000))
    for pet in ipt['PETruth']:
        if pet[2]<1000:
            petruthfull[pet[0]-1,pet[1],pet[2]] += 1
    ipt.close()
    return (petruthfull, EventID)
if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt', help='output')
    psr.add_argument('-m', dest='model', help='model')
    psr.add_argument('-i', dest='ipt', help='problem file')
    args = psr.parse_args()
    inFile = args.ipt
    outFile = args.opt
    modelFile = args.model

    (petruth, eventId) = ReadPETruth(inFile)
    
    model = MLP()
    model.load_state_dict(torch.load(modelFile))
    model.eval()
    
    particleTruth = np.zeros(eventId.shape)
    test_dataset = torch.from_numpy(petruth.reshape(-1,1,30,1000))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for i, wave in enumerate(test_loader):
        wave = wave.float()
        out = model(wave)
        particleTruth[i] = out.cpu().detach().numpy()
    writeSubfile(particleTruth, eventId, outFile)
    print('End write {}'.format(outFile))
    