import numpy as np, h5py, time, argparse, sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import tables

batchSize= 256
begincut = 200
endcut = 700
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)# 16@500 30
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)# stride=1, padding=3) 32@ 1000 30
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)# 64@1000 30
        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)
        #self.norm1 = nn.BatchNorm2d(16)
        #self.norm2 = nn.BatchNorm2d(32)
        #self.norm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*30*125, 1000)
        self.fc2 = nn.Linear(1000, 1)
    def forward(self,x):
        x = F.relu(self.conv1(x))# 16@500 30
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 2))# 32@ 250 30
        x = F.max_pool2d(F.relu(self.conv3(x)), (1, 2))# 64@ 125 30
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))# 64*30*125
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
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
    petruthfull = np.zeros((EventID.shape[0],30,endcut-begincut))
    for pet in ipt['PETruth']:
        if pet[2]>=begincut and pet[2]<endcut:
            petruthfull[pet[0]-1,pet[1],pet[2]-begincut] += 1
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
    model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
    model.eval()
    
    particleTruth = np.zeros(eventId.shape)
    test_dataset = torch.from_numpy(petruth.reshape(-1,1,30,endcut-begincut))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for i, wave in enumerate(test_loader):
        wave = wave.float()
        out = model(wave)
        particleTruth[i] = out.cpu().detach().numpy()
    particleTruth[particleTruth<0.3] = 0
    particleTruth[particleTruth>0.7] = 1
    writeSubfile(particleTruth, eventId, outFile)
    print('End write {}'.format(outFile))
    