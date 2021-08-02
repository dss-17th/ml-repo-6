# Painting vs Photo Binary Classification 프로젝트

## 1. 개요

### 1-1. 프로젝트 목적
Kaggle의 photo vs painting Dataset을 통해 사진과 그림 뷴류해낼 수 있는 이진분류모델을 만드는 것이 주 목적이며, 다양한 CNN Network의 모델링을 통해 가장 졿은 성능을 가진 모델을 도출하는 것입니다.

### 1-2. 프로젝트 목표
- 사진과 그림간 차이점 및 특성 파악
- 다양한 CNN Network 모델 구축 및 최종 모델 선정
- Module을 이용한 코드의 효율성 추구
- 한정된 리소스에서 메모리 관리

### 1-3. 기술적 목표
- Tensorflow Keras를 이용한 네트워크 구성 및 학습
- Tensorflow callback 함수를 이용한 각 모델의 최적의 weights 도출
- Image Generator를 통한 모델 학습(고용량이미지의 경우)
- Pandas, OpenCV를 이용한 데이터 전처리
- BGR, HSV, Hist벡터화 등의 다양한 이미지 전처리를 통한 Input Dataset 생성
- matplotlib, seaborn을 통한 EDA 및 결과 데이터 Visualization

### 1-4. 설명 및 데이터셋
**본 프로젝트의 모델링 과정은 GPU 사용을 위해 Colab에서 진행하였습니다.**

데이터 : Kaggle Dataset
- Train Dataset : 7041 datas (photo: 3528, painting: 3513)
- Validation Dataset : 3010 datas (photo: 1505, painting: 1505)

<br/>
<img width="550" alt="raw_images" src="https://user-images.githubusercontent.com/80459520/126477266-486ad4c4-b3a7-45d8-8b0a-e982c32499fa.png">
<br/>

### 1-5. 팀구성
- 송강 ([GitHub](https://github.com/rivels))
  - Modeling
  - README 작성
- 이승주 ([GitHub](https://github.com/aeea-0605))
  - EDA 및 Modeling
  - README 작성

---
---

## 2. 결론

#### 초기 모델링 및 스토리라인
<img width="1006" alt="스크린샷 2021-08-02 오후 7 27 45" src="https://user-images.githubusercontent.com/80459520/127847489-1b42a270-5b4c-42f0-9259-a8c4f50fa99f.png">

LeNet부터 ResNet Network와 ResNet에서는 각각의 하이퍼 파라미터 튜닝을 통해 총 8번의 모델링을 한 결과 Train Accuracy와 Loss는 각각 1과 0으로 수렴한다.

그러나 Valiation Accuracy는 Train Accuracy와의 격차가 심할 뿐더러 크게 튀는 경우도 존재하고, Loss같은 경우는 Epoch가 진행될수록 점점 증가하기에 신뢰성 있는 예측모델이라고 판단할 수 없기에 Callback 함수를 통한 최적의 가중치를 도출하는 방법으로 모델링 진행.

<br/>

#### <Validation Score를 통한 최종 Network 및 Model 선정>
<img width="1094" alt="result" src="https://user-images.githubusercontent.com/80459520/126478081-b5504af4-3210-4cf5-99ba-036192ff5b39.png">

Efficient Network의 Max Accuracy weights와 Min Loss weights의 예측결과
정확도는 90.5%, 예측 실패한 횟수는 286 으로 가장 좋은 성능을 보였음

**최종모델 : Efficient Network의 Max Accracy, Min Loss Model**

<br/>

---
---

<br/>

## 3. 과정
Efficient Network에 대한 모델링 과정(다른 네트워크들도 대부분 동일한 과정으로 진행)

> ###  1. import module
```
from drive.MyDrive.datas.module.preprocess import *
from drive.MyDrive.datas.module.setting_tf import *
from drive.MyDrive.datas.module.visualization import *
```
- 모델링 및 시각화를 위해 필요한 모듈 import 
   >(모듈 설명 : 맨아래 Code Explanation 부분에 존재)

<br/>

> ### 2. Load Dataset 및 모델링을 위한 Input Dataset 생성
```
# raw dataset 가져오기
dataset = get_dataset() 

# dataset을 train, validation으로 split, 모든 이미지 동일하게 resize한 후 랜덤셔플
datas = split_train_valid_df(dataset=dataset, img_size=224, shuffle=True)

# image와 label 데이터셋으로 세분화
X_train, y_train, X_valid, y_valid = split_X_y_dataset(datas=datas)
```

<br/>

> ### 3. MinMaxscaling and Load CNN Network
```
X_train = X_train / 255.0
X_valid = X_valid / 255.0
```
```
name = "efficientnet"
base_model = load_base_model(name, input_shape=(224, 224, 3), trainable=False)

model = make_network(base_model, name)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
```
- efficient Network를 불러온 후 setting_tf의 make_network함수를 통한 모델 구성

<br/>

> ### 4. Callback Function Setting
```
monitor_ls = ["val_accuracy", "val_loss"]
callbacks = setting_callback("efficientnet", monitors=monitor_ls)
```
- monitor할 대상을 list형식에 넣어준 후 setting_tf의 setting_callback 함수를 통해 weight 저장을 위한 콜백함수 세팅

<br/>

> ### 5. 모델 학습
```
history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                    validation_data=(X_valid, y_valid), verbose=1,
                    callbacks=callbacks)
```
- 콜백함수를 통해 val_accuracy가 Max일 때, val_loss가 Min일 떄 두 경우의 weights를 저장

<br/>

> ### 6. history 및 predict Visualization
```
make_scores_graph(history=history)
```
<img width="720" alt="fit_history" src="https://user-images.githubusercontent.com/80459520/126494909-d1ccad68-48ce-4e24-a53c-5ef44c151a09.png">

- model fitting history의 시각화

```
weight = "/content/drive/MyDrive/datas/model_result/efficientnet_loss.h5"
show_predict(model, weight, X_valid, y_valid)
```
<img width="704" alt="predict" src="https://user-images.githubusercontent.com/80459520/126495209-39a3c781-49d4-44b8-b606-7389da81161f.png">

- Validation Dataset에서 16개의 비복원추출 랜덤샘플링을 통한 Predict Visualization

---
<br/>

# 💡 제언 및 한계점
```
- Train, Validation Dataset의 두 Class의 비율이 5:5이고 사진을 그린 그림, 그림을 찍은 사진들이 있음에도 불구하고 최종 선정 모델의 Accuracy가 90.5% 라면 준수를 예측결과라고 생각합니다.

- 데이터의 양이 많았다면 좀 더 성능 좋은 모델이 도출될 것으로 생각됩니다.

- Train Dataset과 Validation Dataset 간에 픽셀값 간의 특징 외에 다른 특징(같은 작가의 이미지로 판단, 특정 문구로 인한 판단 등) 들이 존재하는지 파악하는 것도 결과의 신뢰성을 높이는 방법 중 하나라고 생각합니다. 
```

<br/>

# Code Explanation
- Module
  - > [preprocess.py](https://github.com/dss-17th/ml-repo-6/blob/main/module/preprocess.py) : Load Dataset 및 modeling을 위한 Input Dataset으로 가공
  - > [setting_tf.py](https://github.com/dss-17th/ml-repo-6/blob/main/module/setting_tf.py) : Load Network 및 콜백함수 세팅
  - > [visualization.py](https://github.com/dss-17th/ml-repo-6/blob/main/module/visualization.py) : 모델링 학습 history, predict 결과 시각화

- EDA
  - > [image_EDA.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/image_EDA.ipynb) : 데이터셋 구성 및 이미지에 대한 EDA Notebook

- Modeling
  - > [MobileNet.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/MobileNet.ipynb) : MobileNet에 대한 Modeling Notebook
  - > [ResNet.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/ResNet.ipynb) : ResNet에 대한 Modeling Notebook
  - > [Inception_ResNet.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/Inception_ResNet.ipynb) : Inception_ResNet에 대한 Modeling Notebook
  - > [InceptionV3.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/InceptionV3.ipynb) : Incaption_V3에 대한 Modeling Notebook
  - > [EfficientNet.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/EfficientNet.ipynb) : EfficientNet에 대한 Modeling Notebook
  - > [others_modeling.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/others_modeling.ipynb) : 초기 모델 및 Hist vector화 후 DenseNet 모델링
- 최종 모델 선정
  - > [select_best_model.ipynb](https://github.com/dss-17th/ml-repo-6/blob/main/select_best_model.ipynb) : 5개의 Network 중 가장 Validation Score가 높은 모델을 선정하는 Notebook
