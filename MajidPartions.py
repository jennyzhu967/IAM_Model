import os

train_read_path = "C:/Users/ericz/Handwriting-Transformers/HCS Data/HCS Dataset December 2021/extracted move boxes/val_data.txt"
train_write_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Majid_val_data.txt"

file_path = "C:/Users/ericz/Handwriting-Transformers/HCS Data/HCS Dataset December 2021/extracted move boxes/"

f = open(train_read_path, "r")

# ft = open(train_write_path, 'r+')
# ft.truncate(0)

for x in f:
    with open(train_write_path, "a") as f:
        path = os.path.join(file_path, x)
        # print(path)
        f.write(path)
