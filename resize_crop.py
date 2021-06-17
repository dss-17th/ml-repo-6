import os
import pickle
import cv2


train_dir = os.path.join(os.getcwd(), 'datas/train')
valid_dir = os.path.join(os.getcwd(), 'datas/valid')

t_painting_dir = os.path.join(train_dir, 'painting')
t_photos_dir = os.path.join(train_dir, 'photos')
v_painting_dir = os.path.join(valid_dir, 'painting')
v_photos_dir = os.path.join(valid_dir, 'photos')

t_paintings = os.listdir(t_painting_dir)
t_photos = os.listdir(t_photos_dir)
v_paintings = os.listdir(v_painting_dir)
v_photos = os.listdir(v_photos_dir)

t_paintings_ls = [os.path.join(t_painting_dir, data) for data in t_paintings]
t_photos_ls = [os.path.join(t_photos_dir, data) for data in t_photos]
v_paintings_ls = [os.path.join(v_painting_dir, data) for data in v_paintings]
v_photos_ls = [os.path.join(v_photos_dir, data) for data in v_photos]
# print("train :", len(t_paintings_ls), len(t_photos_ls))
# print("valid :", len(v_paintings_ls), len(v_photos_ls))

# resize and crop > save to pickle
datasets = [t_paintings_ls, t_photos_ls, v_paintings_ls, v_photos_ls]

total_datas = []
for dataset in datasets:
    datas = []
    for i in range(len(dataset)):
        img = cv2.imread(dataset[i])
        resize_img = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)
        crop_img = resize_img[:400, :]
        datas.append(crop_img)
    total_datas.append(datas)
    
with open('datas/train/resize_painting.pkl', 'wb') as f:
    pickle.dump(np.array(total_datas[0]), f)
    
with open('datas/train/resize_photos.pkl', 'wb') as f:
    pickle.dump(np.array(total_datas[1]), f)
    
with open('datas/valid/resize_painting.pkl', 'wb') as f:
    pickle.dump(np.array(total_datas[2]), f)
    
with open('datas/valid/resize_photos.pkl', 'wb') as f:
    pickle.dump(np.array(total_datas[3]), f)