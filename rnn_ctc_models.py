
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental.preprocessing import StringLookup
#from tensorflow.keras.preprocessing.sequence import pad_sequences


# =================================================================
# RECURRENT MODELS FOR HANDWRITTEN TEXT RECOGNITION
# =================================================================
# https://keras.io/examples/vision/captcha_ocr/



# =================================================
# CTC LOSS LAYER (final layer to compute loss)
# =================================================

class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


# =================================================
# RNN-CTC MODELS
# =================================================

# returns two models: one for training, one for prediction
def build_model_1(hidden_params,
		max_width=200, height=50, vocab_size=10):

    # model parameters
    conv_dim1, conv_dim2, dense_dim, lstm_dim1, lstm_dim2 = hidden_params
    
    # Inputs to the model
    input_img = layers.Input(shape=(max_width, height, 1), name="image", dtype="float32")
    labels = layers.Input(name="target", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(conv_dim1, (3, 3), activation="relu",
    	kernel_initializer="he_normal", padding="same", name="Conv1" )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(conv_dim2, (3, 3), activation="relu",
        kernel_initializer="he_normal", padding="same", name="Conv2" )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((max_width // 4), (height // 4) * conv_dim2)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    # process each 'slice' of length (height // 4) * conv_dim2
    x = layers.Dense(dense_dim, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(lstm_dim1, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_dim2, return_sequences=True, dropout=0.25))(x)

    # Output layer
    output = layers.Dense(vocab_size + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    ctcloss = CTCLayer(name="ctc_loss")(labels, output)

    # Define the model
    model_train = tf.keras.models.Model( inputs=[input_img, labels], outputs=ctcloss, name="ocr_model_v1" )
    model_pred = tf.keras.models.Model( inputs=input_img, outputs=output )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model_train.compile(optimizer=opt)
    return model_train, model_pred




# ==================================================================