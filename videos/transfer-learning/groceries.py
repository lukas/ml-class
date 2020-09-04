from subprocess import call
import os
from urllib.request import urlretrieve
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

dataset_url = "http://aisdatasets.informatik.uni-freiburg.de/" \
              "freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"

def download():
    print("Downloading dataset.")
    urlretrieve(dataset_url, "freiburg_groceries_dataset.tar.gz")
    print("Extracting dataset.")
    call(["tar", "-xf", "freiburg_groceries_dataset.tar.gz", "-C", "."])
    os.remove("freiburg_groceries_dataset.tar.gz")
    print("Done.")

    
def load_data():
    if (not os.path.exists("images")):
        download()
        
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    class_names = []
    category_num = 0
    
    for category in sorted(os.listdir("images")):
        class_names.append(category)
        count = 0
        for ii, img in enumerate(sorted(os.listdir(os.path.join("images", category)))):
            
            if ii % 2:
                continue
            
            if (not img.endswith(".png")):
                continue
                
            x = load_img(os.path.join("images", category, img),target_size=(197, 197))
            if count < 10:
                x_test.append(img_to_array(x))
                y_test.append(category_num)
            else:
                x_train.append(img_to_array(x))
                y_train.append(category_num)
            count += 1
        category_num += 1
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test)), class_names
    
