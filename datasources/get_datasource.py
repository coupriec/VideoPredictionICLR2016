#!/usr/local/bin
import os

datafolder = 'ucf101'

os.system("""
cd %s
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e UCF101.rar
wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist/* .
rmdir ucfTrainTestlist
"""%(datafolder))

files = os.listdir(datafolder)
classes = set()
for f in files:
    if f.find('.avi') != -1:
        cl = f[:f.find('_g')]
        assert(cl[:2] == 'v_')
        cl = cl[2:]
        classes.add(cl)

for cl in classes:
    os.mkdir('%s/%s'%(datafolder, cl))

for f in files:
    if f.find('.avi') != -1:
        cl = f[:f.find('_g')]
        assert(cl[:2] == 'v_')
        cl = cl[2:]
        os.system('mv %s/%s %s/%s'%(datafolder, f, datafolder, cl))

        os.system('mv %s/HandStandPushups %s/HandstandPushups'%(datafolder, datafolder))
