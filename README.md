# Painting vs Photo Binary Classification 프로젝트

## 1. 개요
<br/>

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

### 1-4. 데이터셋 및 설명
Kaggle Dataset
- Train Dataset : 7041 datas (photo: 3528, painting: 3513)
- Validation Dataset : 3010 datas (photo: 1505, painting: 1505)

<br/>
<img width="550" alt="raw_images" src="https://user-images.githubusercontent.com/80459520/126477266-486ad4c4-b3a7-45d8-8b0a-e982c32499fa.png">
<br/>
<br/>

### 1-5. 팀구성
- 송강 ([GitHub](https://github.com/rivels))
  - README 작성
- 이승주 ([GitHub](https://github.com/aeea-0605))
  - README 작성

---
---

## 2. 결론
<br/>

#### <Validation Score를 통한 최종 Network 및 Model 선정>
<img width="1094" alt="result" src="https://user-images.githubusercontent.com/80459520/126478081-b5504af4-3210-4cf5-99ba-036192ff5b39.png">

Efficient Network의 Max Accuracy weights와 Min Loss weights의 예측결과
정확도는 90.5%, 예측 실패한 횟수는 286 으로 가장 좋은 성능을 보였음
> 최종모델 : Efficient Network의 Max Accracy, Min Loss Model

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
- preprocess : Load Dataset 및 modeling을 위한 Input Dataset으로 가공
- setting_tf : Load Network 및 콜백함수 세팅
- visualization : 모델링 학습 history, predict 시각화

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

> ### 6. history 밒 predict Visualization
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

<br/>

### Modeling 현황

<img width="1004" alt="스크린샷 2021-07-11 오후 11 46 08" src="https://user-images.githubusercontent.com/80459520/125199669-3499b500-e2a2-11eb-8490-59b47ff228e8.png">


### Accuracy 및 Loss Visualization

<img width="988" alt="스크린샷 2021-07-11 오후 11 46 39" src="https://user-images.githubusercontent.com/80459520/125199716-82162200-e2a2-11eb-8212-74b90f260690.png">

<img width="970" alt="스크린샷 2021-07-11 오후 11 46 53" src="https://user-images.githubusercontent.com/80459520/125199724-922e0180-e2a2-11eb-84b9-b25326c24de8.png">

<img width="498" alt="스크린샷 2021-07-11 오후 11 47 04" src="https://user-images.githubusercontent.com/80459520/125199730-99550f80-e2a2-11eb-8200-a70e59865ae8.png">

<img width="541" alt="스크린샷 2021-07-11 오후 11 47 13" src="https://user-images.githubusercontent.com/80459520/125199731-9fe38700-e2a2-11eb-84f8-af2b25dadda0.png">

<img width="716" alt="스크린샷 2021-07-11 오후 11 47 21" src="https://user-images.githubusercontent.com/80459520/125199752-abcf4900-e2a2-11eb-9883-8fa6bfe88e6a.png">
