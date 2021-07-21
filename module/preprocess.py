import os, glob
import zipfile
import numpy as np
import pandas as pd
import cv2
import warnings
warnings.simplefilter("ignore")


def get_dataset(only_unzip=False):
  zip_path = os.path.join(os.getcwd(), "drive/MyDrive/datas/")
  filename = "archive_ml.zip"
  local_zip = zip_path + filename

  zip_ref = zipfile.ZipFile(local_zip, 'r')

  zip_ref.extractall('/content/Dataset')
  zip_ref.close()
  
  if only_unzip:
      return print("complete unzip")

  path = "/content/Dataset/"
  dataset = {"image_path": [], "status": [], "where": []}

  for where in os.listdir(path):
    if where in ["Raw Data", "test"]:
      continue

    for status in os.listdir(os.path.join(path, where)):
      for image in glob.glob(path+where+"/"+status+"/"+"*.jpg"):
        dataset["image_path"].append(image)
        dataset["status"].append(status)
        dataset["where"].append(where)

  return pd.DataFrame(dataset)


def resize_img(image, size=256):
  if image.shape[0] > size:
    image = image[:size, :]
  
  if image.shape[1] > size:
    image = image[:, :size]
  
  if image.shape[0] < size:
    image = cv2.resize(image, (size, image.shape[1]))
  
  if image.shape[1] < size:
    image = cv2.resize(image, (image.shape[0], size))

  return image
  
  
def cvt_hist_vec(image, get_segment=False):
    img_b = cv2.calcHist(image, [1], None, [256], [0, 256])
    img_g = cv2.calcHist(image, [2], None, [256], [0, 256])
    img_r = cv2.calcHist(image, [3], None, [256], [0, 256])
    
    if get_segment:
        return img_b, img_g, img_r
    
    return np.vstack([img_b, img_g, img_r]).flatten()


def split_train_valid_df(dataset, img_size=224, shuffle=True):
    datas = {"train_df": [], "valid_df": []}
    
    for idx in range(len(dataset)):
      img = cv2.imread(dataset['image_path'][idx])
      
      resized_img = resize_img(img, size=img_size)
    
      if dataset['status'][idx] == "photos":
        data = [resized_img, 1]
      else:
        data = [resized_img, 0]
    
      if dataset['where'][idx] == 'train':
        datas['train_df'].append(data)
      else:
        datas['valid_df'].append(data)
    
    datas['train_df'] = np.array(datas['train_df'])
    datas['valid_df'] = np.array(datas['valid_df'])
    
    if shuffle:
        np.random.shuffle(datas['train_df'])
        np.random.shuffle(datas['valid_df'])
    
    return datas
    
    
def split_X_y_dataset(datas, get_train=True, get_valid=True):
    X_train, y_train = [], []
    X_valid, y_valid = [], []
    
    for key, value in datas.items():
      for img in value:
        if key == "train_df":
          X_train.append(img[0])
          y_train.append(img[1])
        else:
          X_valid.append(img[0])
          y_valid.append(img[1])
    
    result = []
    if get_train:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        result.append(X_train)
        result.append(y_train)
    
    if get_valid:
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
        result.append(X_valid)
        result.append(y_valid)
    
    return result
    
    


    
    
    
    
