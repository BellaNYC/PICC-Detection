import glob
import os
import shutil

png = glob.glob('png/*png')
exp = glob.glob('april_experiment/image/*png')

training_set = []
for p in exp:
    training_set.append(os.path.basename(p))

for p in png:
    file = os.path.basename(p)
    if file not in training_set:
        shutil.copy2(p, 'test/')
    else:
        print('file in train set')
