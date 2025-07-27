import os

INPUT_DATASET = "datasets/original"

BASE_PATH = "datasets/idc"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# splitting the dataset into percentages
TRAIN_SPLIT = 0.8   # 80% (70% for training and 10% for validation)
VAL_SPLIT = 0.1     # 10% of the 80%
