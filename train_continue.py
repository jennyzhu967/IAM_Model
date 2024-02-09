import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import torch
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from model import ChessNetwork
from configs import ModelConfigs
import pandas as pd

dataset, vocab, max_len = [], set(), 0 #dataset is  alist, vocab is a set, max_len is an int

path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Majid_train_data.txt"

# Keep the same training and testing split for each trial
train_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Augmentation FineTune/ClassifierOnly/train.csv"
df_train = pd.read_csv("C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Augmentation FineTune/ClassifierOnly/train.csv").values.tolist()

test_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Augmentation FineTune/ClassifierOnly/val.csv"
df_test = pd.read_csv("C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Augmentation FineTune/ClassifierOnly/val.csv").values.tolist()

words = open(path, "r").readlines()

for line in tqdm(words):
    position = line.find('png')
    rel_path = line[:position+3]
    label = line[position+3:].rstrip("\n")
    label = label.strip()

    dataset.append([rel_path, label])
    max_len = max(max_len, len(label))

    for i in label:
        vocab.update(list(i))

vocab = sorted(vocab)
# print(vocab) -> ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']

# Save vocab and maximum text length to configs
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Split the dataset into training and validation sets
# train_dataProvider, test_dataProvider = data_provider.split(split = 0.9)
train_dataProvider= DataProvider(
    dataset = df_train,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

test_dataProvider = DataProvider(
    dataset = df_test,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), # comment out when using augmented data
    ]

# Create (pre_trained) network structure, loss function, and optimizer
pre_trained_network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
weights_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/MajidChessPartitions/model.pt"
# NOTE: weights path without graident tracking

# weights_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/ClassifierOnly/model.pt"
# NOTE: weights path for fine-tuning with gradient trakcing
pre_trained_network.load_state_dict(torch.load(weights_path))

# Create a chess model that takes in the parameters of the pretrained model layers but has different output shape
# chess_network = ChessNetwork(num_chars = len(configs.vocab), preTrained = pre_trained_network, classifier = True)

pre_trained_network.fineTune(feature_extract=True) # Turn of gradient tracking (freeze the layers)
# Debugging Code:
# for x in pre_trained_network.pretrained:
#     print(f"Parameters: \n {list(x.parameters())}")
#     for p in x.parameters():
#         print(f"grad: {p.requires_grad}")

loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(pre_trained_network.parameters(), lr=configs.learning_rate)

# create callbacks
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

model = Model(pre_trained_network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])

# train the chess_network model
model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=200, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))