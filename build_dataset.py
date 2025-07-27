from cancernet import config
from imutils import paths
import random, shutil, os

originalPaths=list(paths.list_images(config.INPUT_DATASET))
random.seed(7)
random.shuffle(originalPaths)

index=int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths=originalPaths[:index]
testPaths=originalPaths[index:]

index=int(len(trainPaths)*config.VAL_SPLIT)
valPaths=trainPaths[:index]
trainPaths=trainPaths[index:]

# splitting the dataset into training, validation, and testing
# 70% for training, 10% for validation, and 20% for testing
datasets=[("training", trainPaths, config.TRAIN_PATH),
          ("validation", valPaths, config.VAL_PATH),
          ("testing", testPaths, config.TEST_PATH)
]

# this nested for loop will create each path for training, validation, and testing for the model
# it then seperates the path by 0 or 1 (negative or positive for breast cancer) in each of the paths
for (setType, originalPaths, basePath) in datasets:
    print(f'Building {setType} set')

    if not os.path.exists(basePath):
        print(f'Building directory {basePath}')
        os.makedirs(basePath)

    for path in originalPaths:
        file=path.split(os.path.sep)[-1]
        label=file[-5:-4]

        labelPath=os.path.sep.join([basePath,label])
        if not os.path.exists(labelPath):
                print(f'Building directory {labelPath}')
                os.makedirs(labelPath)

        newPath=os.path.sep.join([labelPath, file])
        shutil.copy2(path, newPath)
