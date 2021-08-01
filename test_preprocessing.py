# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]


import os
import re
import pathlib
#from glob import glob # module for finding pathnames
from time import time

# read image sizes without loading the images
#import PIL
import struct

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


# execute file from python interpreter
# exec(open('name.py').read())

# =================================================================
# =================================================================
# TESTS WITH THE TENSORFLOW DATASET OBJECT
# TO READ A SET OF DATA FILES IN FOLDER
# =================================================================
# =================================================================

# - prepare dataset object where input data requires preprocessing
# - we first create a dataset with file paths
# - then, we map a preprocessing function to the dataset

# preprocessings (text standardization and vectorization, and feature normalization)
# depends on global characteristics of the data and thus is not done in the dataset structure

# tf.data.Dataset object
#  - object that iterate over a dataset (so as the full dataset does not need to fit into memory)
#  - apply preprocessing to dataset

# ---------------------------------------------------------------------

# choose dataset
datapath = "C:\\datasets\\captcha_images_v2" # https://github.com/AakashKumarNain/CaptchaCracker
#datapath = "C:\\datasets\\IAM" # https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

# dataset configuration
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32

# preprocessing settings
HEIGHT = 80 # standardized image height
# maximum width/height ratio to the full padded image
# (in IAM, the longest is 40.04, and the second longest is 34.9)
MAX_ASPECT_RATIO = 4 # for IAM: 35, for captcha 4
MAX_WIDTH = HEIGHT*MAX_ASPECT_RATIO
MAX_TEXT_LEN = 5 # for IAM (80 or 93); for captcha (5)
VOCAB_SIZE = 19 # for IAM: 79; for captcha 19

# initialize function of look up table
char_to_num = None
num_to_char = None


# ----------------------------------------------
#   R E A D I N G   F I L E S  
# ----------------------------------------------

# of PNG files
# https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
def get_image_size(fname):

	with open(fname, 'rb') as fhandle:
		head = fhandle.read(32)
		if len(head) != 32:
			return
		check = struct.unpack('>i', head[4:8])[0]
		if check != 0x0d0a1a0a:
			return
		width, height = struct.unpack('>ii', head[16:24])
	return width, height

# # functions to verify input image sizes
# def image_shape(path):
# 	raw = tf.io.read_file(path)
# 	image = tf.image.decode_png(raw, channels=1)
# 	return tf.shape(image)
# def get_image_shape(inputs):
# 	img_ds = tf.data.Dataset.from_tensor_slices(inputs)
# 	x = img_ds.map(image_shape)
# 	return x

# -------------------------------------------------------------------

# input path is string
def captcha_pair_samples(datapath):

	path = pathlib.Path(datapath)

	image_paths1 = list(path.glob("*.png")) # list of windowspath objects
	image_paths2 = list( map(str,image_paths1) ) # list of strings

	# labels texts (are image file names)
	target_text = [img.split('\\')[-1].split(".png")[0] for img in image_paths2]

	# get present characters with set() method
	characters = sorted( set(char for label in target_text for char in label))

	# shuffle data
	num_samples = len(image_paths2)
	p = np.random.permutation(num_samples)
	image_paths2 = [ image_paths2[i] for i in p ]
	target_text = [ target_text[i] for i in p ]
	return image_paths2, target_text, characters


# for IAM Dataset
# return list of every single training sample (sample audio and corresp. transcription)
def iam_pair_samples(datapath):
	input_path = []
	target_text = []
	with open(os.path.join(datapath, "ascii/lines.txt"), encoding="utf-8") as f:
		for i,line in enumerate(f):
			if line[0] != '#':
				comp = line.split(' ')

				code = comp[0]
				thresh = comp[2] # thresh to binary image
				text = comp[-1]

				path1, path2, _ = code.split('-')
				image_path = datapath + f'\\lines\\{path1}\\{path1}-{path2}\\{code}.png'
				text = re.sub('\|', ' ', text)
				
				if os.path.exists(image_path):
					input_path.append(image_path)
					target_text.append(text[:-1]) # last element in a '\n'
				else:
					print(f'Image file of code {fileid} did not found!')
	
	print('Read paths and texts.')
	
	# get present characters with set() method
	characters = sorted(set(char for label in target_text for char in label))
	print('Detected unique characters.')

	# shuffle data
	num_samples = len(input_path)
	p = np.random.permutation(num_samples)
	input_path = [ input_path[i] for i in p ]
	target_text = [ target_text[i] for i in p ]
	print('Shuffled data.')

	return input_path, target_text, characters


# ----------------------------------------------
#   F E A T U R E   E X T R A C T I O N
# ----------------------------------------------

#@tf.function
# def path2img(path):
# 	# read image from file
# 	raw = tf.io.read_file(path)
# 	image = tf.image.decode_png(raw, channels=1)
# 	# compute new image width (to keep aspect ratio)
# 	width = tf.cast( tf.round(HEIGHT*tf.shape(image)[1]/tf.shape(image)[0]), tf.int32 )
# 	# convert to float32 and resize
# 	image = tf.image.convert_image_dtype(image, tf.float32)
# 	image = tf.image.resize(image, [HEIGHT, width])
# 	image = tf.squeeze(image, axis=-1)
# 	return tf.transpose( tf.pad(image,[[0,0],[0,MAX_WIDTH-width]], constant_values=1.0) )

@tf.function
def join_image_label(image_path, text):
	# read image from file
	raw = tf.io.read_file(image_path)
	image = tf.image.decode_png(raw, channels=1)
	# compute new image width (to keep aspect ratio)
	width = tf.cast( tf.round(HEIGHT*tf.shape(image)[1]/tf.shape(image)[0]), tf.int32 )
	# convert to float32 and resize
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.resize(image, [HEIGHT, width])
	image = tf.squeeze(image, axis=-1)
	image = tf.transpose( tf.pad(image,[[0,0],[0,MAX_WIDTH-width]], constant_values=1.0) )
	# convert text to num and pad
	tokens = char_to_num(tf.strings.unicode_split(text, input_encoding="UTF-8"))
	tokens = tf.pad( tokens, [[0,MAX_TEXT_LEN-tf.shape(tokens)[0]]], constant_values=VOCAB_SIZE)
	return {"image": image, "target": tokens}

def create_ocr_dataset(inputs, targets, batch_size=32, train_test_split=0.8):

	ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
	ds = ds.map(join_image_label).batch(batch_size)
	#ds = ds.prefetch(tf.data.experimental.AUTOTUNE) # for multi-core processing
	# (3) split into train/valid datasets
	num_train = int( len(ds)*train_test_split )
	return ds.take(num_train), ds.skip(num_train)

# convert token to string (input is a tensor)
def tokens2text(tokens):
	# remove null charcters
	seq = tokens[tokens!=VOCAB_SIZE]
	# convert to sequence of chars
	seq = num_to_char(seq+1)
	# join and decode string
	return tf.strings.join(seq).numpy().decode()



# =====================================================
print('\n(1) Captcha Images Dataset')
# =====================================================

# (a) read data (and shuffle)

t1 = time()
input_paths, target_texts,characters = captcha_pair_samples(datapath)
t2 = time(); print('a) Load and arrange transcription pairs: %.3fs'%(t2-t1))

NUM_SAMPLES = len(input_paths)
TRAIN_BATCHES = int( NUM_SAMPLES*(TRAIN_TEST_SPLIT)/BATCH_SIZE )
NUM_TRAIN = TRAIN_BATCHES * BATCH_SIZE
NUM_VALID = NUM_SAMPLES-NUM_TRAIN
print(f'   number of transcription pairs: {NUM_SAMPLES}')
print(f'   number of training samples: {NUM_TRAIN}')
print(f'   number of validation samples: {NUM_VALID}')
print('   Ordered vocabulary:',characters)

# (b) create dataset object and preproccess

# Mapping characters to integers
char_to_num = StringLookup(vocabulary=characters, num_oov_indices=0, mask_token=None)
# Mapping integers back to original characters
num_to_char = StringLookup(vocabulary=characters, mask_token=None, invert=True)

t1 = time()
train_ds, valid_ds = create_ocr_dataset(input_paths, target_texts,
	batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT)
t2 = time(); print('b) Arrange dataset object: %.3fs'%(t2-t1))

# (c) examples

# print('c) Dataset examples')
# t1 = time()
# for pair in train_ds.take(10):
# 	feat, target = pair
# 	t2 = time()
# 	print(' ', feat.shape, '%.3fs'%(t2-t1))
# 	t1=t2

# (3) examples
print('c) Dataset examples')
t1 = time()
for pair in train_ds.take(5):
	t2 = time()
	print(' ', pair["image"].shape, '%.3fs'%(t2-t1))
	t1=t2


# # =====================================================
# print('\n(2) IAM Dataset')
# # =====================================================

# # (a) read data (and shuffle)

# t1 = time()
# input_paths, target_texts, characters = iam_pair_samples(datapath)
# t2 = time()
# NUM_SAMPLES = len(input_paths)
# print('a) Load and arrange transcription pairs: %.3fs'%(t2-t1))
# print('   Number of transcription pairs:',NUM_SAMPLES)
# print('   Ordered vocabulary:',characters)

# # (b) verify aspect ratio of input images
# # and remove images with aspect ratio higher than specified

# t1 = time(); print("b) Verify images aspect ratio")
# to_del = []
# for i,p in enumerate( list(zip(input_paths,target_texts)) ):
# 	path, sent = p
# 	width, height = get_image_size(path)
# 	if width/height >= MAX_ASPECT_RATIO or len(sent)>MAX_TEXT_LEN:
# 		to_del.append(i)
# 	if i%1000==0:
# 		print("  ", i)

# print("   removed images: ", len(to_del))
# for i in sorted(to_del, reverse=True):
# 	del input_paths[i]
# 	del target_texts[i]
# t2 = time(); print('   elapsed time: %.3fs'%(t2-t1))

# NUM_SAMPLES = len(input_paths)
# TRAIN_BATCHES = int( NUM_SAMPLES*(TRAIN_TEST_SPLIT)/BATCH_SIZE )
# NUM_TRAIN = TRAIN_BATCHES * BATCH_SIZE
# NUM_VALID = NUM_SAMPLES-NUM_TRAIN
# print(f'   number of transcription pairs: {NUM_SAMPLES}')
# print(f'   number of training samples: {NUM_TRAIN}')
# print(f'   number of validation samples: {NUM_VALID}')


# # (c) create dataset object and preproccess

# # Mapping characters to integers
# char_to_num = StringLookup(vocabulary=characters, num_oov_indices=0, mask_token=None)
# # Mapping integers back to original characters
# num_to_char = StringLookup(vocabulary=characters, mask_token=None, invert=True)

# t1 = time()
# train_ds, valid_ds = create_ocr_dataset(input_paths, target_texts,
# 	batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT)
# t2 = time(); print('b) Arrange dataset object: %.3fs'%(t2-t1))

# # (3) examples
# print('c) Dataset examples')
# t1 = time()
# for pair in train_ds.take(5):
# 	t2 = time()
# 	print(' ', pair["image"].shape, '%.3fs'%(t2-t1))
# 	t1=t2


# # =====================================================
# print('\n(x) IAM Tests: verify images and texts max sizes')
# # =====================================================

# # verify input image sizes
# heis = np.zeros(NUM_SAMPLES, dtype=np.int32)
# wids = np.zeros(NUM_SAMPLES, dtype=np.int32)
# rats = np.zeros(NUM_SAMPLES)
# for i,p in enumerate(input_paths):
# 	raw = tf.io.read_file(p)
# 	image = tf.image.decode_png(raw, channels=1)
# 	heis[i] = image.shape[0]
# 	wids[i] = image.shape[1]
# 	rats[i] = image.shape[1]/image.shape[0]
# 	if i%1000==0:
# 		print(i)

# num_tokens = np.zeros(NUM_SAMPLES, dtype=np.int32)
# for i,p in enumerate(target_texts):
# 	num_tokens[i] = len(p)
# 	if i%1000==0:
# 		print(i)

# plt.figure()
# plt.plot(num_tokens)



# =================================================
print('\n(xxx) Sample Plots')
# =================================================

plt.figure(1); plt.clf()
plt.subplot(3,1,1)
plt.imshow(tf.transpose(pair["image"][0,:,:]), cmap=plt.cm.jet, aspect='auto')
plt.title( tokens2text(pair["target"][0,:]) )
plt.subplot(3,1,2)
plt.imshow(tf.transpose(pair["image"][1,:,:]), cmap=plt.cm.jet, aspect='auto')
plt.title(tokens2text(pair["target"][1,:]))
plt.subplot(3,1,3)
plt.imshow(tf.transpose(pair["image"][2,:,:]), cmap=plt.cm.jet, aspect='auto')
plt.title(tokens2text(pair["target"][2,:]))
plt.show()






# ==================================================================