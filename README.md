# Handwritten Text Recognition

In this test, I implement a recurrent neural network (RNN) to recognize a sequence of handwritten characters. Unlike the conventional optical character recognition (OCR) approaches, in which each character is recognized separately, requiring therefore a previous step where  the characters are partitioned, with RNN we can retrieve the sequence of characters without explicitly partitioning into character units.

In particular, I use the **Connectionist Temporal Classification (CTC) algorithm**, that jointly computes the loss function and estimates the best alignment between the input sequence and the target label sequence. This algorithm was firstly proposed for speech recognition [1], and subsequently was applied to the handwritten text recognition task [2], and resembles the forward-backward and the Viterbi algorithms used in classical speech recognition tasks.

I implemented the whole process (preprocessing, training and prediction steps) for a simple Captcha reader task (Captcha Cracker database [3]), and still implemented only the preprocessing step for a more sofisticated task in recognizing the IAM dataset [4]. The codes were based on [5-7].

## About the model

The model used in this task has 6 layers:

 - Conv2D layer, 32 kernels of size 3x3, unit stride, followed by ReLU activation and 2x2 Max Pooling;
 - Conv2D layer, 62 kernels of size 3x3, unit stride, followed by ReLU activation and 2x2 Max Pooling;
 - Dense layer, followed by ReLU activation and dropout, with 64 units that process separately each data slice from left to right;
 - Bidirectional LSTM layer with 128 units;
 - Bidirectional LSTM layer with 64 units;
 - Dense layer with 20 units (there are 19 valid tokens) with softmax activation.

## Sample result

The training and validation loss along the training process is depicted below.

<img src="https://github.com/ryuuji06/handwritten-text-recognition/blob/main/images/ex_hist.png" width="500">

The figure below shows a test input image and the corresponding output of the network (probabilities of each token). The last token (cyan) is the null character token, inherent of the CTC algorithm, and encodes no actual character. Note that the model detected correctly the characters present in the captcha image in the correct order. However, the CTC algorithm to duplicate the predicted characters in a row (interval in which the probability of this character iss higher than the others), and thus I still need to perform some posterior handling to remove duplicates.

<img src="https://github.com/ryuuji06/handwritten-text-recognition/blob/main/images/ex_captcha.png" width="800">

## References

[1] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. Proceedings of the 23rd International Conference on Machine Learning (ICML'06), p.369-376, 2006

[2] A. Graves and J. Schmidhuber. Offline handwriting recognition with multidimensional recurrent neural networks. Proceedings of the 21st International Conference on Neural Information Processing Systems (NIPS'08), p.545-552, 2008

[3] https://github.com/AakashKumarNain/CaptchaCracker

[4] IAM database. https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

[5] Keras tutorial. https://keras.io/examples/vision/captcha_ocr/

[6] https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5

[7] https://towardsdatascience.com/faq-build-a-handwritten-text-recognition-system-using-tensorflow-27648fb18519
