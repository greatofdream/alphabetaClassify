#import moxing as mox
import os
import h5py, numpy as np
import argparse

#mox.file.shift('os', 'mox')
#obs_root = 's3://deeplearning-zaq/test'
obs_root=''
fileprefix = '{}/data/ghostPlay'.format(obs_root)
outputprefix = '{}/output/ghostPlay/'.format(obs_root)
file = ''

psr = argparse.ArgumentParser()
psr.add_argument('-f', dest='file', help='input file name')
psr.add_argument('-i', dest='idir', help='input dir')
psr.add_argument('-o', dest='odir', help='output dir')
args = psr.parse_args()
file = args.file
fileprefix = args.idir
outputprefix = args.odir
begincut = 200
endcut = 700
if file =='':
    print('please input file name')
    exit(1)
with h5py.File(fileprefix + file, 'r') as ipt, h5py.File(outputprefix + file, 'w') as opt:
    print('{}:{}'.format(fileprefix + file,list(ipt.keys())))
    petruthfull = np.zeros((ipt['ParticleTruth'].shape[0], 30, 1000-(endcut - begincut)))
    particleType = ipt['ParticleTruth']['Alpha'][:]
    for i in ipt['PETruth']:
        if i[2]>=begincut and i[2]<endcut:
            petruthfull[i[0]-1,i[1],i[2]-begincut] += 1
    opt.create_dataset('PETruth', data=petruthfull, compression='gzip')
    opt.create_dataset('ParticleType', data=particleType, compression='gzip')
    print('finish write;PETruth.shape:{};ParticleType.shape:{}'.format(opt['PETruth'].shape,opt['ParticleType'].shape))


