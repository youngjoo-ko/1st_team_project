import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
import PIL
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2 # 224 x 224 이미지에 작동하도록 설계되어 있음. 가벼움
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # keras mobilenet은 특정종류의 입력전처리가 필요.
from tensorflow.keras.optimizers import RMSprop 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
# CNN (Convolution Neural Network) - 합성곱 신경망 (기존에는 데이터에서 지식을 추출해 학습) 
# CNN 은 데이터 특징을 추출하여 특징들의 패턴을 파악하는 구조 - Convolution과정과 Pooling 과정을 통해 진행
# Convolution - 데이터의 특징을 추출하는 과정으로 데이터에 각 성분의 인접 성분들을 조사해 특징을 파악하고 파악한 특징을 한장으로 도출시키는 과정.
# 여기서 도출된 장을 Convolution Layer라고 함. 이과정은 하나의 압축 과정이며 파라미터의 갯수를 효과적으로 줄여줌
# Pooling - Convolution 과정을 거친 레이어의 사이즈를 줄여주는 과정. 단순히 데이터의 사이즈를 줄여주고, 노이즈를 상쇄시키며 미세한 부분에서 일관적인
# 특징을 제공 - 보통 정보추출, 문장분류, 얼굴인식에서 사용됨

# Fully Connected Layer만으로 구성된 인공신경망으 ㅣ입력 데이터는 1차원(배열) 형태로 한정됨
# 한장의 컬러사진은 3차원 데이터. 배치모드에 사용되는 여러장의 사진은 4차원 데이터
# 사진 데이터로 Fully Connected신경망을 학습시켜야 할 경우에, 3차원 사진 데이터를 1차원으로 평면화 시켜야 한다.
# 사진 데이터를 평면화 시키는 과정에서 공간 정보가 손실됨
# 결과적으로 이미지 공간 정보 유실로 인한 정보 부족으로 인공 신경망이 특징을 추출 및 학습이 비효율적이고 정확도를 높이는데 한계가 있음
# 이미지의 공간 정보를 유지한 상태로 학습이 가능한 모델이 바로 CNN이다.

# CNN은 기존 Fully Connected Neural Network와 비교하여 다음과 같은 차별을 갖는다.
# 각 레이어의 입출력 데이터의 형상 유지
# 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
# 복수의 필터로 이미지의 특징 추출 및 학습
# 추출한 이미지의 특징을 모으고 강화하는 Pooling레이어
# 필터를 공유 파라미터로 사용하기 때문에, 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적음.
################################################################################################
# https://m.blog.naver.com/msnayana/220776380373
# 단순하게 표현하면
# CNN은 신경망에 기존의 필터기술을 병합하여 신경망이 2차원 영상을 잘 습득할 수 있도록 최적화 시킨 방법(알고리즘)으로
# MLP는 모든 입력이 위치와 상관없이 동일한 수준의 중요도를 갖기에 이를 이용해 fully-connected neural network를 구성하게 되면
# 파라미터의 크기가 엄청나게 커지는 문제가 생기고 이에 대한 해결책으로 탄생한 것이 CNN

# Convolution - 기존 영상처리의 필터 + 신경망 = CNN
# 좀더 세분화하면 convolution과 sub-sampling을 반복하여 데이터량을 줄이고 왜곡시켜 신경망에서 분류케 만든다
# 일반적 모델은 특징추출(컨벌루션) + 분류행위(신경망) = 분류 결과
# 영상처리에서 컨벌루션은 가중치를 갖는 마스크를 이용해서 영상처리를 하는것을 의미
# 입력영상에 마스크를 씌운다음, 입력 영상의 픽셀값과 마스크의 가중치를 각각 곱한후 그합을 출력영상의 픽셀값으로 정하는것
# 영상처리에 사용되는 마스크를 필터, 윈도 또는 커널이라고 한다.

# why CNN?
# DNN의 문제점 : 기본적으로 1차원 형태의 데이터를 사용
# DNN은 2차원 형태의 이미지가 입력값이 되면 이것을 flatten 시켜서 한줄 데이터로 만들어야하는데
# 이 과정에서 이미지의 공간적/지역적(spatial/topological) 정보가 손실됨
# 또한 추상화 과정 없이 바로 연산과정으로 넘어가 버리기 때문에 학습시간과 능률의 효율성이 저하됨
# 이러한 문제점에서부터 고안한 해결책이 CNN - 이미지를 raw input 그대로 받음으로 공간적/ 지역적 정보를
# 유지한채 특성(feature)들의 계층을 빌드업한다. CNN의 중요 포인트는 이미지 전체보다는 부분을 보는것
# 그리고 이미지의 한 픽셀과 주변 픽셀들의 연관성을 살리는것

# CNN은 이미지의 공간 정보를 유지하며 인접 이미지와의 특징을 효과적으로 인식하고 강조하는 방식으로
# 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분으로 구성
# 특징 추출은 filter를 사용하여 공유 파라미터 수를 최소화 하면서 이미지의 특징을 찾는 convolution레이어와 강화하고 모으는
# pooling레이어로 구성
# CNN은 filter의 크기, stride, padding과 pooling크기로 출력 데이터 크기를 조절하고
# 필터의 개수로 출력 데이터의 채널을 결정
# cnn은 같은 레이어 크기의 fully connected Neural network와 비교해 볼때
# 학습 파라미터 양은 20%규모. 은닉층이 깊어질 수록 학습 파라미터의 차이는 더 벌어진다.
# CNN은 fully connected neural network와 비교하여 더작은 학습 파라미터로 더 높은 인식률을 제공


# CNN은 학습 파라미터 수가 매우 작음
# 학습 파라미터가 작고, 학습이 쉽고 네트워크 처리 속도가 빠름
dataset = 'C:\\auto-mask-checkingbot-project\Mask_Checking_Camera\dataset'
imagePaths = list(paths.list_images(dataset))

data = []
labels = []

for i in imagePaths:
    try:
        label = i.split(os.path.sep)[-2]
        labels.append(label)
        image = load_img(i, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
    except PIL.UnidentifiedImageError:
        os.remove(i)
        labels.remove(label)
        print("remove i : ",i)
        print("remove label : ",label)
        pass
print(len(data))
print(len(labels))

data = np.array(data, dtype='float32')
labels = np.array(labels)
# print(data.shape)
# print(data)
# print(labels)
# 라벨에 원핫 인코딩 수행
# one-hot encoding 이란 단 하나의 값만 True이고 나머지는 모두 False인 인코딩을 말함. 즉, 1개만 Hot(True) 나머지는 Cold(False)
# e.g> [0,0,0,0,1] -? 5번째만 1이고 나머지는 0 행렬을 자주 사용하는 연산에서는 4와 같은 스칼라 값보다 [0,0,0,0,1]과 같은 행렬이 자주 사용됨
lb = LabelBinarizer()
# sklearn - fit_transform함수 : 저장된 데이터의 평균을 0으로 표준편차를 1로 바꾸어줌
labels = lb.fit_transform(labels)
# keras - to_categorical : 클래스 백터를 이진 클래스 행렬로 변환.
labels = to_categorical(labels)
# print(labels)

# 모델을 학습하고 그 결과를 검증하기 위해서는 원래의 데이터를 training, validation, testing 의 용도로 나눠야함
# scikit-learn패키지중 model_selection에는 데이터 분할을 위한 train_test_split 함수가 있음
# arrays : 분할 시킬 데이터를 입력
# test_size : 테스트 데이터 셋의 비율(float)이나 갯수(int) defalt = 0.25
# train_size : 학습 데이터 셋의 비율 
# random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값.
# shuffle : 셔플 여부 설정 default = True
# stratify : 지정한  Data의 비율을 유지. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary set일때 , stratify=Y로 설정하면
# 나누어진 데이터셋들도 0과 1을 각각 25%, 75% 로 유지한채 분할.
train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size = 0.20, random_state = 20, stratify=labels)
# print(train_X.shape)
# print(train_Y.shape)
# print(test_X.shape)
# print(test_Y.shape)

# ImageDataGenerator : 데이터 증가를 위한 훈련 이미지 생성
# ImageDataGenerator 클래스를 통해 객체를 생성할 때 마다 파라미터를 전달해주는 것을 통해 데이터의 전처리를 쉽게 할수 있고
# 또 이 객체의 flow_from_directory 메소드를 활용하면 폴더 형태로된 데이터 구조를 바로 가져와서 사용할 수 있다.
# rotation_range : 정수, 무작위 회전의 각도 범위
# zoom_range :  부동소수점 혹은  [하한, 상한] 무작위 줌의 범위. 부동소수점인 경우[하한,상한] = [1-zoom_range, 1+zoon_range]
# width_shift_range : 부동소수점, 1D 형태의 유사배열 혹은 정수
# - 부동소수점 : < 1 인경우  전체 가로 넓이에서의 비율, >= 1 인 경우 픽셀의 개수.
# - 1D형태의 유사배열 : 배열에서 가져온 무작위 요소
# - 정수 : (-width_shift_range, +width_shift_range) 사이 구간의 픽셀 개수
# - width_shift_range = 2 인 경우 유효값은 정수인 [-1, 0, 1]로, width_shift_rnage = [-1, 0, 1] 와 같은 반면
# - width_shift_range = 1.0 인 경우 유효값은 [-1.0, +1.0]의 반개 구간사이 부동소수점이다.
# width_shift_range : 상기 width_shift_range와 같다
# shear_range : 부동소수점. 층밀리기의 강도 (도 단위의 반시계 방향 층 밀리기 각도)
# horizontal_flip : boolean, 인풋을 무작위로 가로로 뒤집는다.
# vertical_flip : boolean, 인풋을 무작위로 세로로 뒤집는다.
# fill_mode : constant, nearest, reflect, wrap 중 하나. 디폴트 값은 'nearest' 인풋 경계의 바깥 공간은 다음의 모드에 따라 다르게 채워짐
# 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
# 'nearest': aaaaaaaa|abcd|dddddddd
# 'reflect': abcddcba|abcd|dcbaabcd
# 'wrap': abcdabcd|abcd|abcdabcd
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range= 0.15, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
# print(aug)

# fine-tunning (미세조정)을 위해 MobilenetV2
# weights= 'imagenet' : 로드할 가중치 파일의 경로(기본값=imagenet)
# include top : 네트워크 상단에 완전 연결 계층을 포함할지 여부를 나타내는 bool값 (기본 = True)
# input_tensor : layers.Input()모델의 이미지 입력으로 사용할 선택적 keras텐서(출력). input_tensor는 여러 다른 네트워크 간에 입력을 공유하는데 유용 (기본값 = None)
baseModel = MobileNetV2(weights='imagenet', include_top = False, input_tensor=Input(shape=(224, 224, 3)))

# baseModel.summary()
# 훈련데이터 상단에 배치 될 모델의 헤드부분 구성
headModel = baseModel.output
# AveragePooling2D : 공간데이터에 대한 평균 풀링 작업
# pool_size : 2개 정수의 정수 또는 튜플, 축소 할 인수(수직, 수평). (2,2) 두공간 차원에서 입력을 절반으로 줄인다. 정수를 하나만 지정하면 두 차원에 동일한 창 길이가 사용됨.
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# Flatten : 레이어 병합, 입력을 평평하게 한다. 배치 크기에 영향을 주지 않음.
# 참고 : 입력이(batch)피쳐 축없이 모양이 지정되는 경우 병합은 추가 채널 치수를 추가하고 출력모양은 (batch,1)
headModel = Flatten(name = 'Flatten')(headModel)
# Dense : 조밀 한 층, 규칙적으로 조밀하게 연결된 Neural Network
headModel = Dense(128, activation = 'relu')(headModel)
# Dropout : 신경망 모델을 만들때 생기는 문제중 'overfitting'이라는 문제가 있음.
# overfitting은 train data를 너무 잘 사용하여 train accuracy는 높으나 test data를 넣었을 때에는 test accuracy가 낮게 나오는 것이다.
# 이를 해결하기 위한 방법이 Dropout
# 드랍아웃은 완전 연결에서 일부 연결을 의도적으로 끊는 것
# 신경망 연결에서 일부를 덜 학습시켜서 과적합을 방지하는 것.
# 드랍아웃 적용시 설정하는 인자 3가지
# rate : 드랍 아웃을 적용할 확률 값 : 0~ 1 (0~ 100%)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = 'softmax')(headModel) # 전결합층(fully-conntected layer)추가 (Model.add()를 통해 추가할수 있음)

# Model & Sequential
# 보통 keras를 이용해 신경망 설계를 할 때, Sequential을 사용(선형)하고 시퀀셜사용시 분석 모델을 층으로쌓듯 매우 쉽게 구축이 가능
# 보다 복잡한 구조를 만들려면 함수api를 이용하여 모델을 작성하는 방식을 알아야 한다.

model = Model(inputs = baseModel.input, outputs = headModel)

# 네트워크의 기본 레이어를 고정함
# 기본 레이어의 가중치는 역 전파 프로세스중에 업데이트 되지 않지만
# 헤드레이어 가중치는 조정
for layer in baseModel.layers:
    layer.trainable = False

model.summary()
# 한번의 epoch 는 인공 신경망에서 전체 데이터 셋에 대해 forward pass/backward pass 과정을 거친 것을 말함. 즉 전체 데이터 셋에 대해 한번 학습을 완료한 상태
# 신경망에서 사용되는 역전파 알고리즘(bakcpropagation algorithm)은 파라미터를 사용하여 입력부터 출력까지의 각 계층의  weight을 계산하는 과정을 거치는 순방향패스
# (forward pass), forward pass 를 반대로 거슬러 올라가며 다시 한 번 계산 과정을 거쳐 기존의 weight를 수정하는 역방향 패스(backward pass)로 나뉨
# 이 전체 데이터 셋에 대해 해당 과정 (forward pass + backward pass)이 완료되면 한번의 epoch 가 진행됐다고 볼 수 있음.
# 지금은 epochs 20으로  데이터를 20번 사용해서 학습을 거침. epoch값이 너무 작다면 underfitting 크면 overfitting이 발생
learning_rate = 0.0001
Epochs = 30
BS = 256

# Optimizer(최적화)
# Optimization은 학습속도를 빠르고 안정적이게 해줌.

# RMSprop 알고리즘
# 그라디언트 제곱의 이동(할인) 평균 유지 이평균의 루트로 그라디언트를 나눈다.
# 중심버전은 추가로 그라디언트의 이동 평균을 유지하고 해당 평균을 사용하여 분산을 추정한다.
# rho: 히스토리 / 다가오는 그라디언트에 대한 할인 요소, 기본 0.9
# momentum: 스칼라 또는 스칼라 tensor, 기본은 0.0
# epslion: 수치적 안정성을 위한 작은 상수
# decay : 업데이트마다 적용되는 학습의 감소율

# Adam 알고리즘
# 목적함수의 최소값을 찾는 알고리즘으로 Momentum과 RMSprop의 기본 아이디어를 합친 알고리즘.
# Momentum term에 RMSprop에서 등장한 지수이동 평균을 적용하ㅣ고  RMSprop과 같은 방식으로 변수들이 update하는 방향과 크기는 Adaptive벡터에
# 의해서 결정된다 (소득 재분배)
# 주요 장점은 stepsize가 gradient의 rescaling에 영향을 받지 않는다는 것.
# gradient가 커져도 stepsize는 bound되어 있어서 어떠한 obejctive function을 사용한다 하더라도 안정적으로 최적화를 위한 하강이 가능하다.
# stepsize를 과거의 gradient크기를 참고하여 adapted시킬수 있다.
opt = Adam(lr = learning_rate, decay = learning_rate/Epochs) #learning_rate/Epochs
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

# batch size는 한번의 batch마다 주는 데이터 샘플의 size. 여기서 batch(보통 mini-batch로 표현)는 나눠진 데이터 셋을 뜻하며 iterations는 epoch를 나누어서 실행하는 횟수
# 메모리의 한계와 속도 저하 때문에 대부분의 경우에는 한번의 epoch에서 모든 데이터를 한꺼번에 집어넣울 수는 없다.
# 그래서 데이터를 나누어서 주게 되는데 이때 몇번 나누어서 주는가를 iteration 각 iteration마다 주는 데이터 사이즈를 batch_size라고 한다

# mini batch
H = model.fit(
    aug.flow(train_X, train_Y, batch_size = BS),
    steps_per_epoch = len(train_X)//BS,
    validation_data = (test_X, test_Y),
    validation_steps = len(test_X)//BS,
    epochs = Epochs
)

# model.save_weight('autocheckingWeight.h5')
model.save('C:\\auto-mask-checkingbot-project\\Mask_Checking_Camera\\modelpack\\autochecking_model8.h5')

# argmax, argmin => one-hot 인코딩에 사용하는 함수

# link =  https://www.tensorflow.org/api_docs/python/tf/math/argmax
# argmax => 최대 값의 인덱스값 반환 argmin => 최소 값의 인덱스값 반환

predict = model.predict(test_X, batch_size = BS)
predict = np.argmax(predict, axis = 1)

# 결과값
# precision(정밀도) : model이 True라고 분류한 것중에서 실제 True인것의 비율 -> ppv(positive predictive value)
# recall(재현율) : 실제 True인 것 중에서 모델이 True라고 예측한것의 비율
# precision은 모델의 입장에서, recall은 실제 정답(data)의 입장에서 정답을 정답이라고 맞춘 경우를 바라봄.

# Accuracy(정확도) : 위 두 지표는 모두 True를 True라고 옳게 예측한 경우만 다룸
# accuracy는 False를 False라고 예측한 경우도 옳은 경우이므로 이 경우도 고려하는 지표.
# 만약 날씨를 예측하는 경우 한달 동안이 특정 기후에 부합하여 비오는 날이 흔치 않다고 생각했을때 이 경우에는 해당 data의 domain이 불균형 하게 되므로
# 맑은 것을 예측하는 성능은 높지만, 비가 오는 것을 예측하는 성능은 매우 낮을 수 밖에 없다. 이를 보환할 지표가 필요.

# F1 Score(조화평균) : precision과 recall의 조화 평균
# F1score는 label이 불균형 구조일 때 모델의 성능을 정확하게 평가할 수 있으며, 성능을 하나의 숫자로 표현할 수 있다.

# macro avg : 단순평균
# weighted avg : 각 클래스에 속하는 표본의 개수로 가중평균
print(classification_report(test_Y.argmax(axis = 1), predict, target_names = lb.classes_))

# plot the training loss and accuracy

N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Adam batch256-epoch30-lr0.001-data30000")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('C:\\auto-mask-checkingbot-project\\Mask_Checking_Camera\\modelpack\\finalmodel_graph_adam.png')
