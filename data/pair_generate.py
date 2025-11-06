import random
import os

random.seed(19)
scale = 4

root_path=os.path.dirname(os.path.dirname(os.getcwd()))
base_dir = os.path.join(root_path, "The file address where your data is stored")

hr_dir = os.path.join(base_dir, "HR")
lr_dir = os.path.join(base_dir, "LR")

name_lists_hr = sorted([name for name in os.listdir(hr_dir) if name.endswith(".tif") ])
name_lists_lr = sorted([name for name in os.listdir(lr_dir) if name.endswith(".tif") ])

dataset_list = [os.path.join(hr_dir, i) + " " + \
 os.path.join(lr_dir, j) + "\n" for i,j in zip(name_lists_hr, name_lists_lr)]

train_list = random.sample(dataset_list, 1850) 
val_list =  [i for i in dataset_list if i not in train_list]
val_list = random.sample(val_list, 462) 

with open(base_dir + "/train_" + str(scale) + "x.txt", "w") as f:
    for name in train_list:
        f.write(name)

with open(base_dir +  "/val_" + str(scale) + "x.txt", "w") as f:
    for name in val_list:
         f.write(name)
