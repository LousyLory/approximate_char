import glob
import tqdm

all_files = glob.glob('/home/ray/annotations/current/*')
all_files = [x.rpartition('/')[-1] for x in all_files]
all_files = [x.strip('.npy') for x in all_files]

count = 0
for singlefile in tqdm.tqdm(all_files):
	with open('parser_'+str(count)+'.sh', 'w') as f:
		f.write('#!/bin/bash\n')
		f.write('python archan_matcher_VGG16.py '+singlefile+'\n')
	count += 1

