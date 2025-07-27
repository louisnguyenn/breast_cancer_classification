from tensorflow.keras.models import Sequential          # we use sequential API and SeparableConv2D to implement depthwise convolutions
from tensorflow.keras.layers import SeparableConv2D     # (width and height of the image, and its depth (number of colour channels in each image))
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

# this class/function will be responsbie for the model's deep learning of the breast cancer images
# it analyzes complex patterns in each image and learns from them to detect breast cancer
# this is called a CNN (convolutional neural network), where in deep learning the model will find patterns by analyzing thousands
# of images, as opposed to machine learning where we manually train the model with images, deep learning analyzes and observes it on its own
# by feeding it thousands of information. you can think of the CNN as like the human brain, where there are many networks feeding each other
# information.
class CancerNet:
    @staticmethod
    # width and height of the image, depth of the colour channels, and number of class (0 or 1, negative or positive) which would be 2 in this case
    def build(width, height, depth, classes):
        model = Sequential()
        shape = (height, width, depth)
        channelDim = -1

        if K.image_data_format() == "channels_first":
            shape = (depth, height, width)
            channelDim = 1

        model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the model
        return model
