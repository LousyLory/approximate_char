import numpy as np
import sys
import glob
import os
import copy
import tqdm

def check_availability(main_dir):
	if not os.path.isdir(main_dir):
        	os.mkdir(main_dir)

character = sys.argv[1]

main_dir = 'character_annotations'
check_availability(main_dir)
sub_dir = main_dir+'/'+character
check_availability(sub_dir)

# grab all files
all_files = glob.glob('save_dir/*')
filenames = copy.copy(all_files)
filenames = [x.rpartition('/') for x in all_files]

for i in tqdm.tqdm(range(len(all_files))):
	annotations = np.load(all_files[i]).item()
	dict_ = {}
	count = 0
	for j in annotations.keys():
		if annotations[j]['name'] == character:
			dict_[count] = annotations[j]
		else:
			pass
		count += 1
	np.save(os.path.join(sub_dir, filenames[i]), dict_)
