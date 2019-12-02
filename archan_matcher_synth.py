# write a dynamic algorithm to compute the scores for assigning
# a character a bounding box. It should be noted that each word
# is associated with a bounding box and as such we just have to
# compute the linear optimal assignment. Let score to be just S
# for now (we can define this later). Let space be #. Each word
# is then represented as #A#r#c#h#a#n#, e.g., Archan. The cost 
# of assigning a space is uniform irrespective of the width.

#--------------------------ALGORITHM---------------------------#
'''
cost_of_#: c1
Input_region: word_bounding_box
Input_word: word_annotation
len_word: len(Input_word)
cost_matrix = zeros(2*len_word+1, 2*len_word+1)

update rule:

if id is odd: cost_now = cost_of_#
else: cost_now = score(between `j` and `n`) (`n` is the length)

cost_matrix[id,j] = max(cost_matrix(id-1,0:j))+score

'''
#--------------------------------------------------------------#

import sys
import random
#sys.path.append('../finetune_alexnet_with_tensorflow/')

#main_dir = '/mnt/nfs/work1/elm/ray/finetune_alexnet_with_tensorflow_case_sensitive/'

#main_dir = '/mnt/nfs/work1/elm/ray/alexnet_with_map_data/'

main_dir = '/home/ray/finetune_alexnet_with_tensorflow/'

sys.path.append(main_dir)

import matplotlib.image as mpimg
from scipy import ndimage

import numpy as np
import cv2
import math
import os
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator
import string
from strlocale import BasicLocale

# set random seed
#tf.set_random_seed(1234)

# set up object for string's unicode conversion
lc = BasicLocale()

# global constants
#VGG_MEAN = np.asarray([123.68, 116.779, 103.939])
VGG_MEAN = np.asarray([103.939, 116.779, 123.68])
sess = None
score = None

#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# necessary placeholders for tensorflow sessions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
#y = tf.placeholder(tf.float32, [None, 36])
keep_prob = tf.placeholder(tf.float32)

################################################################
'''helper function for cropping'''
################################################################
def distance(x1, x2):
    return int(math.hypot(x2[0] - x1[0], x2[1] - x1[1]))

def orientation(x1, x2):
    if float(x2[0] - x1[0]) == 0:
        if x2[1] - x1[1] > 0:
            return -90
        else:
            return 90
    else:
        return math.degrees(math.atan2(x2[1] - x1[1], x2[0] - x1[0]))

#######################################################################
'''Given a fulcrum, annotation and image returns a image crop'''
#######################################################################
def get_crop(Img, V, fulcrum):
    '''
    get a good crop of the region around/bounded by V
    '''
    V = np.asarray(V)
    rowmin = int(min(V[:,1]))
    rowmax = int(max(V[:,1]))
    colmin = int(min(V[:,0]))
    colmax = int(max(V[:,0]))
    Img_out = Img[rowmin:rowmax+1, colmin:colmax+1, :]
    fulcrum = np.asarray(fulcrum) - np.asarray([colmin, rowmin])
    return Img_out, fulcrum, np.asarray([colmin, rowmin])

#######################################################################
'''Rotate an image given a a pivot and orientation and other details'''
#######################################################################
def rotateImage(img, angle, pivot, height, width):
    '''
    rotate the image given the above information
    '''
    padX = [600+int(img.shape[1] - pivot[0]), 600+int(pivot[0])]
    padY = [600+int(img.shape[0] - pivot[1]), 600+int(pivot[1])]
    imgP = np.pad(img, [padY, padX, [0, 0]], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    centerRow = int(imgR.shape[0]/2)
    centerCol = int(imgR.shape[1]/2)
    imgR = imgR[centerRow-height+1:centerRow+1, centerCol : centerCol+width-1, :]
    # crop to a fixed size
    '''
    if imgR.shape[1] > imgR.shape[0]:
        ratio = float(188)/imgR.shape[1]
    else:
        ratio = float(188)/imgR.shape[0]
    imgR = cv2.resize( imgR, (0,0), fx=ratio, fy=ratio )
    '''
    return imgR

def reshape_image(image):
	return cv2.resize(image, dsize=(227, 227), interpolation=cv2.INTER_LINEAR)

################################################################
'''point rotations'''
################################################################
def rotate(xy, theta):
    #print xy
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )

################################################################
'''point translations (if necessary)''' 
################################################################
def translate(xy, offset):
    return xy[0] + offset[0], xy[1] + offset[1]

def make_data(image, character):
	# make the data format correct for Tensorflow
	try:
                _id = string.ascii_uppercase.index(character)
                #print _id
        except:
		try:                        
                        _id =  string.ascii_lowercase.index(character)+26
                except:
                        _id = range(10).index(int(character))+52
                #print _id

        #label = _id
	label = 1

        # evaluating params
        batch_size = 1

        # Network params
        num_classes = 2

        image = np.float32(image)-np.float32(VGG_MEAN)
        image = np.expand_dims(image, axis=0)

        # generate one hot variable
        #one_hot = tf.one_hot(label, num_classes)
        one_hot = np.zeros((num_classes))
        one_hot[label] = 1
        one_hot = np.expand_dims(one_hot, axis=0)
        one_hot = np.float32(one_hot)
	
	return image, one_hot, _id

#############################################################################################
'''Loads one tensorflow session'''
#############################################################################################
def load_session(batch_size, num_classes, character):
	# configuring settings
	#tf.reset_default_graph()
        # Initialize model
        model = AlexNet(x, keep_prob, num_classes, [])

        # Link variable to model output
        score = tf.nn.softmax(model.fc8)

        #initialize store model
        saver = tf.train.Saver()

	# initialize a session
	sess = tf.Session()
	saver = tf.train.import_meta_graph(os.path.join(main_dir, 'tmp/finetune_alexnet/chkpt/'+str(character)+'_model_epoch30.ckpt.meta'))
	saver.restore(sess, os.path.join(main_dir,'tmp/finetune_alexnet/chkpt/'+str(character)+'_model_epoch30.ckpt'))
	#sess.run(tf.global_variables_initializer())
        #graph = tf.get_default_graph()
	
	return sess, score

#############################################################################################
'''Given a crop and a character computes the score'''
#############################################################################################
def smart_characterness_score(image, character, sess, score):
	# computes the characterness score
	image, one_hot, label = make_data(image, character)
	#score_out = sess.run(score, feed_dict={x: image, y: one_hot, keep_prob: 1.})
	#print label
	score_out = score
	score_out = score_out.eval(feed_dict={x: image, keep_prob: 1.}, session=sess)#y: one_hot, keep_prob: 1.}, session=sess)
	#print score_out[0, np.argmax(score_out)], np.argmax(score_out)
	#return score_out[0, label]
	return score_out[0, 1]

#########################################################################################
'''This is the dynamic thingy'''
#########################################################################################
def find_alignment(image, region, word, cost_of_space):
	## here is the dynamic thing happenning
	# store the beginning column of a character
	# instead of dynamic here i use the rough estimates to fine tune the exact region
	# for the characters
	begin_alignment = []
	# store the ending column of a character
	end_alignment = []

	# basically here you fix the region
	altered_region, shift_matrix, _angle = find_word_image(image, region)
	if altered_region is not None:
		pass
	else:
		print "invalid annotation"
		return None
	# check if image is cropped fine or not
	#cv2.imwrite('./sample.jpg', altered_region)
	# convert word to list
	annotation_to_list = list(word)
	# since in this approach space isn't required I'm going to do away with this
	'''
	annotation = []
	for i in range(len(annotation_to_list)):
		annotation.append('space')
		annotation.append(annotation_to_list[i].lower())
	annotation.append('space')
	'''
	annotation = annotation_to_list
	# removing for case sensitivity
	#annotation = [anots.lower() for anots in annotation_to_list]
	'''
	### also generate the rough estimates
	Things to note: rough estimates generated as :
	for ELon: divide length by 4. For the first extend the right boundary by 1.5 times
	for other extend all boundaries by 1.5. for the last extend the left boundary by 
	1.5 times. that would be enough.	
	'''
	rough_estimates_begin = []
	rough_estimates_end = []
	average_width = altered_region.shape[1] / float(len(annotation))
	for i in range(len(annotation)):
		#print i*average_width - average_width/2
		rough_estimates_begin.append(max(0,i*average_width - average_width/3))
		rough_estimates_end.append(min((i+1)*average_width-1+average_width/3, altered_region.shape[1]-1))
	# convert everything to int
	rough_estimates_begin = [int(x_id) for x_id in rough_estimates_begin]
	rough_estimates_end = [int(x_id) for x_id in rough_estimates_end]
	
	print rough_estimates_begin, rough_estimates_end
	# initialize an emnpty array of scores
	val_matrix = np.zeros((len(annotation), altered_region.shape[1], altered_region.shape[1]))
	# now go ahead and fill up the above matrix
	# loop through all characters
	for _id in range(len(annotation)):
		# loop through all the available columns
		print annotation[_id]
		first_index = rough_estimates_begin[_id]
		last_index = rough_estimates_end[_id]
		########################################################################################################
		# load binary classifier (adding construct for case sensitivity)
        	try:
                	char_id = string.ascii_uppercase.index(annotation[_id])
        	except:
			try:
				char_id =  string.ascii_lowercase.index(annotation[_id])+26
			except:
                		char_id = range(10).index(int(annotation[_id]))+52
		if _id >= 0:
			tf.reset_default_graph()
			# necessary placeholders for tensorflow sessions
			global x
			global keep_prob
			x = tf.placeholder(tf.float32, [1, 227, 227, 3])
			#y = tf.placeholder(tf.float32, [None, 36])
			keep_prob = tf.placeholder(tf.float32)
		sess, score = load_session(batch_size=1, num_classes=2, character=char_id)
		print 'loaded session for character: '+annotation[_id]+' with ID: '+str(char_id)
		########################################################################################################
		for j in range(first_index,last_index-1):
			# loop through all range of widths
			for k in range(j+1,last_index):
				# check if its a space (if yes, assign the same score throughout)
				#print j, k
				#print A[x]
				if annotation[_id] == 'space':
					new_score = cost_of_space
				else:
					# else compute score using classifier
					altered_image = reshape_image(altered_region[:,j:k+1])
					new_score = smart_characterness_score(altered_image, annotation[_id], sess, score)
					#cv2.imwrite(annotation[_id]+'_'+str(new_score)+'.png', altered_image,  [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

				if _id == 0:
					prev_score = 0
				else:
					prev_score = val_matrix[_id-1, begin_alignment[-1], end_alignment[-1]]
				#val_matrix[_id,j,k] = prev_score + new_score
				val_matrix[_id,j,k] = new_score
	
		best_alignment = np.unravel_index(np.argmax(val_matrix[_id,:,:]), [1,val_matrix.shape[1], val_matrix.shape[2]])
		begin_alignment.append(best_alignment[1])
		end_alignment.append(best_alignment[2])
		print begin_alignment[-1], end_alignment[-1], val_matrix[_id, begin_alignment[-1], end_alignment[-1]]
	
	#np.save('val_mat_74k.npy', val_matrix)
	return altered_region, begin_alignment, end_alignment, shift_matrix, _angle, val_matrix	
	#return alignment

###########################################################################################
'''Return a rectangular image given a image and set of vertices'''
###########################################################################################
def find_word_image(image, vertices):
	## here the region is cropped and sent back for character search

	# check if the annotation is a rectangle (polygons can be done 
	# using B-splines)
	#print vertices
	if len(vertices) == 4:
		fulcrum = map(int,vertices[0])
		x2 = vertices[1]
        	x4 = vertices[3]
        	width = int(distance(fulcrum, x2))
        	height = int(distance(fulcrum, x4))
		#print 'height, width', height, width
	        _angle = orientation(fulcrum, x2)
		im, fulcrum, shift_matrix = get_crop(image, vertices, fulcrum)
		#print 'image size: ', im.shape
		extracted_crop = rotateImage(im, _angle, fulcrum, height, width)
		#print 'extracted crop size:', extracted_crop.shape
		#print vertices[0], _angle
	else:
		return None
	return extracted_crop, shift_matrix, _angle

############################################################################################
'''converts the vcertices in horizontal images to actual ones'''
############################################################################################
def reset_vertices(shapes, begin_a, end_a, shift_matrix, _angle, dict_, name):
	_id = len(dict_)
	x = shapes[0]
	y = shapes[1]
	name = list(name)
	for i in range(len(begin_a)):
		# get the vertex in the current transformed form
		vertices = [[begin_a[i], x], [end_a[i], x], [end_a[i], 0], [begin_a[i], 0]]
		print vertices
		# change it world form
		vertices = [rotate(xy, -_angle) for xy in vertices]
		#print 'rotated vertices', vertices
		# transpose
		vertices = np.array(vertices)+shift_matrix
		#print 'shift_matrix', shift_matrix
		dict_[_id+i] = {'vertices': np.array(vertices), 'name': name[i]}
	return dict_

###########################################################################################
'''save individual bounding boxes'''
###########################################################################################
def save_vertice_images(image, begin_a, end_a, count):
	x, y, z = image.shape
	for i in range(len(begin_a)):
                vertices = [[begin_a[i], x], [end_a[i], x], [end_a[i], 0], [begin_a[i], 0]]
		I = plot_vertices(image, [vertices])
		cv2.imwrite('./sample_char_loc_synth'+str(count)+'_'+str(i)+'.jpg', I)
	return None

def plot_vertices(Im, all_predictions):
	I = np.copy(Im)
	shift_matrix = np.array([0,0])
	for i in range(len(all_predictions)):
		predictions = all_predictions[i]
		pt1 = [int(predictions[0][0]), int(predictions[0][1])]
		pt2 = [int(predictions[1][0]), int(predictions[1][1])]
		pt3 = [int(predictions[2][0]), int(predictions[2][1])]
		pt4 = [int(predictions[3][0]), int(predictions[3][1])]
	
		cnt = np.array([pt1, pt2, pt3, pt4])
		cnt += shift_matrix
		cv2.drawContours(I, [cnt], 0, 255)
	
	#cv2.imwrite('colored_'+str(i)+'.jpg', I)
	return I

############################################################################################
'''plot all in images and show'''
############################################################################################
def plot_images(image, dict_):
	#print dict_
	for i in dict_.keys():
		predictions = dict_[i]['vertices']
		#print predictions
		pt1 = [int(predictions[0][0]), int(predictions[0][1])]
	        pt2 = [int(predictions[1][0]), int(predictions[1][1])]
        	pt3 = [int(predictions[2][0]), int(predictions[2][1])]
        	pt4 = [int(predictions[3][0]), int(predictions[3][1])]
            	cnt = np.array([pt1, pt2, pt3, pt4])
		cv2.drawContours(image, [cnt], 0, 255)
	'''
	baseline = old_dict['vertices']
	pt1 = [int(baseline[0][0]), int(baseline[0][1])]
        pt2 = [int(baseline[1][0]), int(baseline[1][1])]
        pt3 = [int(baseline[2][0]), int(baseline[2][1])]
        pt4 = [int(baseline[3][0]), int(baseline[3][1])]
        cnt = np.array([pt1, pt2, pt3, pt4])
        cv2.drawContours(image, [cnt], 0, (0,0,255))
	'''
	return image

###########################################################################################
'''test function to check for a single image'''
###########################################################################################
def function_to_check(image, character):
	q = smart_characterness_score(image, character)
	print q
	return None

############################################################################################
'''function to find a given word in list of annotations'''
############################################################################################
def find_word_from_annotations(randomized_keys, word):
	count  = 0
	for i in randomized_keys:
		if len(annotation[i]['vertices']) == 4 and annotation[i]['name'] is not None and len(annotation[i]['name'])>2:
			if lc.represent(annotation[i]['name']).encode('ascii','ignore') == word:
        			return count, i
			else:
				pass
			count += 1
			
	return None, None

############################################################################################
'''main function to test for a single image'''
############################################################################################
'''
#sess, score = load_session(batch_size=1, num_classes=36)
test_im = cv2.imread('colored.jpg')
test_im = reshape_image(test_im)
function_to_check(test_im, 't')

#'''
############################################################################################
'''The main function'''
#############################################################################################
#'''
# LOAD UP THE SESSION
#sess, score = load_session(batch_size=1, num_classes=36)

# initialize a blank dictionary
dict_ = {}
# THE MAIN LOOP
annotation = np.load('../annotations/current/D0006-0285025.npy').item()
#image = mpimg.imread('../faster-rcnn_backup/DataGeneration/maps_red//D0006-0285025.tiff')
image = cv2.imread('../faster-rcnn_backup/DataGeneration/maps_red//D0006-0285025.tiff')
#randomized_keys = np.random.permutation(annotation.keys())
randomized_keys = annotation.keys()
#count = 35#190
############################################################################################
################################ FIND A WORD IN THE LIST ###################################
#count, i = find_word_from_annotations(randomized_keys, 'St')
#print count, i
############################################################################################
count = 190
#'''
for i in randomized_keys:
	i = 223
	# check if annotations is rectangle and is labelled
	if len(annotation[i]['vertices']) == 4 and annotation[i]['name'] is not None and len(annotation[i]['name'])>2:
		# convert name is ascii versions from unicode
		annotation[i]['name'] = lc.represent(annotation[i]['name']).encode('ascii','ignore')
		# logging prints
		print 'segmenting ',annotation[i]['name']
		# compute the segmentation
		im, begin_a, end_a, shift_matrix, _angle, val_matrix = find_alignment(image,annotation[i]['vertices'], annotation[i]['name'], 1)
		# generate the dictionary
		new_dictionary = reset_vertices(im.shape, begin_a, end_a, shift_matrix, _angle, dict_, annotation[i]['name'])
		# update the original dictionary
		dict_ = new_dictionary
		#break
		cv2.imwrite('./sample_char_loc_synth'+str(count)+'.jpg', im)
		save_vertice_images(im, begin_a, end_a, count)
		count += 1
	if count == 191:
		#191:
		break
	#'''
#new_dictionary = dict_
# plot all generated splits
'''
image = plot_images(image, dict_)

if im is not None:
	print begin_a, end_a
	cv2.imwrite('./sample.jpg', im)
	cv2.imwrite('plot.jpg', image)
else:
	print 'nothing to save.'

# save the image
cv2.imwrite('plot.jpg', image)

np.save('vals.npy', val_matrix)
#'''
pass
