import csv

train_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/trainChess.csv"
path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/trainChessData.txt"

f = open(path, "r")

ft = open(train_path, 'r+')
ft.truncate(0)

for x in f:
    with open(train_path, "a") as f:
        position = x.find('png')
        path = x[:position+3] + "," + x[position+3:]
        f.write(path)
