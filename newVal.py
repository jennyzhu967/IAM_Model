import pandas as pd

val_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/mySplits/validationSet.csv"
new_val_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/valChessData.csv"

ft = open(new_val_path, 'r+')
ft.truncate(0)

set = []
f = open(val_path, "r")

for x in f:
    with open(new_val_path, "a") as f:
        if x not in set:
            set.append(x)
            f.write(x)
