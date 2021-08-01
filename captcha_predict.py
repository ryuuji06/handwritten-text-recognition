# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import re
import pathlib
from time import time

import pickle
import struct

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rnn_ctc_models import build_model_1
#from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt


# execute file from python interpreter
# exec(open('captcha_predict.py').read())

# =================================================================
# =================================================================
# OCR ON captcha_images_v2 WITH CNN-RNN-CTC MODEL
# =================================================================
# =================================================================
# https://keras.io/examples/vision/captcha_ocr/


# ---------------------------------------------------------------------


# preprocessing settings and dataset configuration
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32

HEIGHT = 50 # standardized image height
# width/height ratio to the full padded image
# (in IAM, the longest is 40.04, and the second longest is 34.9)
MAX_ASPECT_RATIO = 4
MAX_WIDTH = HEIGHT*MAX_ASPECT_RATIO
MAX_TEXT_LEN = 5 # for IAM: 93
VOCAB_SIZE = 19 # for IAM: 79

# model parameters
model_params = [32, 64, 64, 128, 64]


# initialize function of look up table
char_to_num = None
num_to_char = None

#folder, model_name = 'test_captcha_03', 'model-94-4.553.h5'
folder, model_name = 'test_captcha_05', 'model-118-0.059.h5'
#folder, model_name = 'test_captcha_xx', 'model-05-16.398.h5'
datapath = "C:\\datasets\\captcha_images_v2"


# ----------------------------------------------
#   R E A D I N G   F I L E S  
# ----------------------------------------------

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


# ----------------------------------------------
#   F E A T U R E   E X T R A C T I O N
# ----------------------------------------------

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
	#ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
	# (3) split into train/valid datasets
	num_train = int( len(ds)*train_test_split )
	return ds.take(num_train), ds.skip(num_train)


# convert token to string (input is a tensor)
def tokens2text(tokens, null_token):
	# remove null charcters
	seq = tokens[tokens!=null_token]
	# convert to sequence of chars
	seq = num_to_char(seq+1)
	# join and decode string
	return tf.strings.join(seq).numpy().decode()


# =====================================================
print('\n(1) Load History and Model')
# =====================================================

# load model loss history
with open(folder+"/hist.pickle", 'rb') as f:
    hist = pickle.load(f)

# load vocabulary
with open(folder+"/vocab.pickle", 'rb') as f:
	vocabulary = pickle.load(f)

# Mapping characters to integers
char_to_num = StringLookup(vocabulary=vocabulary, num_oov_indices=0, mask_token=None)
# Mapping integers back to original characters
num_to_char = StringLookup(vocabulary=vocabulary, mask_token=None, invert=True)

# load model
model_train, model_pred = build_model_1(model_params, max_width=MAX_WIDTH, height=HEIGHT, vocab_size=VOCAB_SIZE)
model_pred.load_weights(folder+'/'+model_name)


# =====================================================
print('\n(2) Test Inference')
# =====================================================

# load list of data
input_paths, target_texts, characters = captcha_pair_samples(datapath)

# select examples and load images
t = np.arange(4,14)
test = tf.data.Dataset.from_tensor_slices((np.array(input_paths)[t], np.array(target_texts)[t]))
test = test.map(join_image_label).batch(10)

for p in test:
	test_data = p

y_pred = model_pred(test_data["image"])
pred_tokens = tf.argmax(y_pred,axis=-1)
pred_codes = []
for i in range(len(t)):
	seq = pred_tokens[i][pred_tokens[i]!=VOCAB_SIZE] # remove null element
	seq = num_to_char(seq+1)
	pred_codes.append(tf.strings.join(seq).numpy().decode())
true_codes = np.array(target_texts)[t]

# =====================================================

plt.figure(); plt.grid()
plt.plot(10*np.log10(hist[0]))
plt.plot(10*np.log10(hist[1]),'o')
plt.legend(['Training loss','Validation loss'])
plt.ylabel('Loss (dB)')
plt.xlabel('Epochs')


i = 0
plt.figure()
plt.subplot(2,1,1)
plt.plot(y_pred[i],'o-'); plt.grid()
plt.title("True: "+true_codes[i]+", predicted: "+pred_codes[i])
plt.ylabel('Output (Token probabilities)')
plt.subplot(2,1,2)
plt.imshow(tf.transpose(test_data["image"][i,:,:]), cmap=plt.cm.jet, aspect='auto')
plt.show()

# compute model output and print

# posterior handling: remove duplicates (common in CTC)



# ==================================================================