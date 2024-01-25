import pandas as pd
import os

train_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/mySplits/trainChessData.csv"
val_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/mySplits/valChessData.csv"

train_DNE = 0
train_duplicate = 0
train_size = 0

val_DNE = 0
val_duplicate = 0
val_size = 0

overlap = 0

set = []
set2 = []

df = pd.read_csv(val_path).values.tolist()
for image_path, label in df:
    val_size +=1
    if image_path in set:
        val_duplicate+=1

    if not os.path.isfile(image_path):
        val_DNE+=1
    set.append(image_path)


df = pd.read_csv(train_path).values.tolist()
DNE = []
for image_path, label in df:
    train_size +=1
    if image_path in set2:
        train_duplicate+=1

    if not os.path.isfile(image_path):
        train_DNE+=1
        DNE.append(image_path)
        continue
    if image_path in set:
        overlap+=1
    set2.append(image_path)

# path2 = 'C:/Users/ericz/Handwriting-Transformers/HCS Data/HCS Dataset December 2021/extracted move boxes/002_0_1_black.png'
# if path2 in set:
#     print(True)

print(f"train DNE: {train_DNE}")
print(f"train Duplicate: {train_duplicate}")
print(f"train Size: {train_size}")
print()
# for i in DNE:
#     print("\n")
#     print(i)

# print(len(DNE))
print(f"val DNE: {val_DNE}")
print(f"val Duplicate: {val_duplicate}")
print(f"val Size: {val_size}")
print()

print(f"overlap: {overlap}")