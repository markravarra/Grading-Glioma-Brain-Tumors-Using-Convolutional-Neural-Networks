# Importing the necessary packages for the Miniature VGG NET
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
    @staticmethod
    # width - the image width dimension
    # height - the image height dimension
    # depth - the depth of the image (number of channels)
    # classes - the number of classes in the data set
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu",
                         input_shape=inputShape))  # 32 filters, 3 x 3 kernel (1ST CONVOLUTIONAL LAYER)
        model.add(MaxPooling2D(pool_size=(2, 2))) # 3 x 3 pool (reducing spatial dimension, depends on input image's dimensions)
        model.add(Dropout(0.25)) # dropouts (Dropout works by randomly disconnecting nodes from the current layer to the next layer.)
        model.add(BatchNormalization(axis=chanDim))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu")) # increase filter size to 64 (2ND CONVOLUTIONAL LAYER)
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu")) # (3RD COONVOLUTIONAL LAYER)
        model.add(MaxPooling2D(pool_size=(2, 2))) # reduce max pooling size to prevent quick reduction of spatial dimensions
        model.add(Dropout(0.25)) # used to reduce overfitting (25%)
        model.add(BatchNormalization(axis=chanDim))
        # Stacking multiple CONV  and RELU  layers together (prior to reducing the spatial
        # dimensions of the volume) allows us to learn a richer set of features.

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (2, 2), padding="same", activation="relu")) # (5TH CONVOLUTIONAL LAYER)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25)) # used to reduce overfitting (25%)
        model.add(BatchNormalization(axis=chanDim))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256, activation="relu")) # FC layer (6TH FULLY CONNECTED LAYER)
        model.add(Dropout(0.5)) # used to reduce overfitting (50%)
        model.add(BatchNormalization())
        # Typically you’ll use a dropout of 40-50% in our fully-connected layers and a dropout with much lower rate,
        # normally 10-25% in previous layers (if any dropout is applied at all).

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        # return the constructed network architecture
        return model
