# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from imutils.video import VideoStream
import tensorflow as tf
import numpy as np

from imutils import paths
import imutils
import time
import cv2 as cv
import os

import serial
import time

# from gpiozero import TonalBuzzer,LED
# from gpiozero.tones import Tone

# ard = serial.Serial('COM6', 9600)
num = 0 # 마스크 유무와 검출 없을시 아두이노로 보낼 값을 담을 변수선언
maskNomask = 0
# 마스크 인식 확인후 LED로 표시 및 부저
# buzzer = TonalBuzzer(21)
# red = LED(14)
# green = LED(15)

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


# ANN (Artificial Neural Network) 인공신경망
# 사람의 신경망 원리와 구조를 모방하여 만든 기계학습 알고리즘. (input, hidden layer, output)
# 히든레이어들의 갯수와 노드의 개수를 구성하는것을 모델링하는 것이라고 함.
# 이를 통해 output값으 ㄹ잘 예측하는 것이 해야 할 일. 은닉층에서 활성화 함수를 사용하여 최적의 weight&Bias를 찾아내는 역할

# DNN (Deep Neural Network) - 은닉층을 2개 이상 지닌 학습 방법

# CNN (Convolution Neural Network) - 합성곱 신경망 (기존에는 데이터에서 지식을 추출해 학습) 
# CNN 은 데이터 특징을 추출하여 특징들의 패턴을 파악하는 구조 - Convolution과정과 Pooling 과정을 통해 진행
# Convolution - 데이터의 특징을 추출하는 과정으로 데이터에 각 성분의 인접 성분들을 조사해 특징을 파악하고 파악한 특징을 한장으로 도출시키는 과정.
# 여기서 도출된 장을 Convolution Layer라고 함. 이과정은 하나의 압축 과정이며 파라미터의 갯수를 효과적으로 줄여줌
# Pooling - Convolution 과정을 거친 레이어의 사이즈를 줄여주는 과정. 단순히 데이터의 사이즈를 줄여주고, 노이즈를 상쇄시키며 미세한 부분에서 일관적인
# 특징을 제공 - 보통 정보추출, 문장분류, 얼굴인식에서 사용됨

# RNN(순환신경망 - Recurrent Neural Network)
# - 반복적이고 순차적인 데이터(Sequential data) 학습에 특화된 인공신경망의 한종류로 내부의 순환구조가 들어있다는 특징을 가짐


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	# blobFromImage = 이미지를 전처리
	# blob이란 동일한 방식으로 전처리 된 동일한 너비, 높이 및 채널 수를 가진 하나 이상의 이미지.
	# 출력값은 4차원 tensor(NCHW) - N은 이미지의 수, C는 채널 수, H는 텐서의 높이, W는 텐서의 너비
	blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
			face = cv.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=256)
		# print("{:.2f} : {:.2f}".format(preds[0][0], preds[0][1]))
	# else: # 얼굴이 없으면 시리얼 통신으로 led를 모두 끄도록한다.
	# 	num = '3'
	# 	num = num.encode("utf-8")
	# 	ard.write(num)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath=os.path.sep.join(['C:\\auto-mask-checkingbot-project\\Mask_Checking_Camera','deploy.prototxt'])
weightsPath=os.path.sep.join(['C:\\auto-mask-checkingbot-project\\Mask_Checking_Camera','res10_300x300_ssd_iter_140000.caffemodel'])

#opencv dnn모듈
# 이미 만들어진 네트워크에서 순방향 실행을 위한 용도로 설계
# 즉, 딥러닝 학습은 기존의 유명한 caffe, tensorflow등 다른 딥러닝 프레임 워크에서 진행하고
# 학습된 모델을 불러와서 실행할 때에는 dnn모듈을 사용하는 방식.

faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet=load_model('C:\\auto-mask-checkingbot-project\\Mask_Checking_Camera\\modelpack\\Bad-model2.h5')

# initialize the video stream
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = cv.VideoCapture(0)
vs.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
vs.set(cv.CAP_PROP_FRAME_HEIGHT, 768)
fps = vs.get(cv.CAP_PROP_FPS)
delay = round(1000/ fps)
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# frame = vs.read()
	# frame = imutils.resize(frame, width=800)
	_, frame = vs.read()
	# frame = cv.flip(frame, 1)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	# print(preds)
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# 예측모델에서 mask가 95퍼센트 이상일때만 Mask로 표기하도록 임계처리
		# label = "Mask" if mask >= 0.95 else "No Mask"
		# label = 0
		# if mask >= withoutMask:
		# 	label = "Mask"
		# 	color = (0, 255, 0)
		# 	buzzer.play(Tone(800.0))
		# 	buzzer.stop()
		# 	# red.off()
		# 	# green.on()
		# else:
		# 	# withoutMask = 1.0
		# 	label = "No Mask"
		# 	color = (0, 0, 255)
		# 	buzzer.play(Tone(440.0))
		# 	buzzer.stop()
			# green.off()
			# red.on()
		label = "Mask" if mask > withoutMask else "withoutMask"
		

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# 마스크 착용유무에 따른 아두이노 시리얼 통신
		# if label == "Mask":
		# 	num = '1'
		# 	num = num.encode('utf-8')
		# 	ard.write(num)
		# else:
		# 	num = '2'
		# 	num = num.encode('utf-8')
		# 	ard.write(num)
		

		# 라벨에 확률 포함
		# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		

		# display the label and bounding box rectangle on the output
		# frame
		cv.putText(frame, label, (startX, startY - 10),
			cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv.imshow("Frame", frame)
	key = cv.waitKey(delay) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv.destroyAllWindows()
# vs.stop()
vs.release()





	