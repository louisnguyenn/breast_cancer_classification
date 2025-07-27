import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# this function will train and evaluate our model

NUM_EPOCHS = 40     # set inital values of number of epochs (tests)
INIT_LR = 1e-2      # learning rate
BS = 32             # batch size

# load and process the data
trainPaths = list(paths.list_images(config.TRAIN_PATH))
lenTrain = len(trainPaths)
lenVal = len(list(paths.list_images(config.VAL_PATH)))
lenTest = len(list(paths.list_images(config.TEST_PATH)))

trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
# classWeight = classTotals.max() / classTotals
classWeight = {i: weight for i, weight in enumerate(classTotals.max() / classTotals)}

trainAug = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

valAug = ImageDataGenerator(rescale=1/255.0)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS
)
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS
)

# build the model
model = CancerNet.build(width=48, height=48, depth=3, classes=2)	# retrieving dimensions of the image (height and weight) and classes 
																	# class 1 = cancer, class 0 = no cancer
opt = Adagrad(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training the model
M = model.fit(
    trainGen,
    steps_per_epoch=lenTrain // BS,
    validation_data=valGen,
    validation_steps=lenVal // BS,
    class_weight=classWeight,
    epochs=NUM_EPOCHS
)

# from here we start to evaluate the model and find the accuracy
print("Now evaluating the model")
testGen.reset()
pred_indices = model.predict(
    testGen,
    steps=(lenTest // BS) + 1
)
pred_indices = np.argmax(pred_indices, axis=1)

print(classification_report(testGen.classes, pred_indices, target_names=list(testGen.class_indices.keys())))

cm = confusion_matrix(testGen.classes, pred_indices)
total = sum(sum(cm))
accuracy = (cm[0,0] + cm[1,1]) / total
specificity = cm[1,1] / (cm[1,0] + cm[1,1])
sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
print(cm)
# here we compute the confusion matrix to get the accuracy, specificity, and sensitivity and display it
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

# here we plot the training loss and accuracy of the model
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
