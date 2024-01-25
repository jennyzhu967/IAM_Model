from PIL import Image
import torchvision.transforms.v2 as transforms
import os

# Sample:
# img_path = "C:/Users/ericz/Handwriting-Transformers/HCS Data/HCS Dataset December 2021/extracted move boxes/086_1_14_white.png"

ground_truth_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/DataAugmentation_Ground_Truth.txt"
ft = open(ground_truth_path, 'r+')
ft.truncate(0)

img_directory_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/Majid_train_data.txt"
directory = "C:/Users/ericz/OneDrive/Desktop/IAM Model/DataAugmentation" # folder to save image to
f = open(img_directory_path, "r")


for x in f:
    """
    Below is a lot of nitty gritty work to get the image labels and path (skip)
    """
    number = 0
    x = x.strip() # removes all the trailing and leading whitespaces (inlcude new line)

    position_path = x.find('png')
    rel_path = x[:position_path+3] # Find the relative path (get rid of ground truth label)
    img = Image.open(rel_path)
    # img.show()

    position_label = x.find('boxes') # returns the first occurence of the word, -1 if not found
    name = x[position_label + 6: position_path+3] 
    # Find the image number [Game #]_[Page #]_[move #]_[black/white] without the ground truth or directory path
    # Ex: 001_0_1_white (Game 1, Page 0, Move 1, Black)

    ground_truth = x[position_path+3:]
    
    augment_10 = "Augment_10_" + name
    orig_image_path = os.path.join(directory, augment_10).replace("\\", "/")
    img.save(orig_image_path) # save the orignal image as Augment 10 (poor planning)

    label = orig_image_path + ground_truth + "\n"
    with open(ground_truth_path, "a") as fx:
        fx.write(label)

    for number in range(10): # loop 10 times, for 10 different augmentated images
        # img = Image.open(rel_path)

        # transform = transforms.RandomChoice(
        # [transforms.RandomRotation(degrees = (-10, 10)), # rotate
        # transforms.RandomAffine(degrees = 0,
        #                         shear = (-15, 15)), # shear
        # transforms.RandomAffine(degrees=0,
        #                         scale = (0.8, 1.2))] # scale (20% smaller, 20% larger) -> Majid paper: -20%, 20%
        # )

        path = "Augment_" + str(number) + "_" + name
        SAVE_PATH = os.path.join(directory, path).replace("\\", "/")
        label = SAVE_PATH + ground_truth + "\n"
        with open(ground_truth_path, "a") as fx:
            fx.write(label)

        # rotated_img = transform(img) # perform transform
        # rotated_img.save(SAVE_PATH) # save image to SAVE_PATH