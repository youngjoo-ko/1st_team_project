import cv2
import numpy as np
import serial
import os
import math
import time


# 아두이노 통신 설정
PORT = 'COM7'
BaudRate = 38400

ARD = serial.Serial(PORT, BaudRate)

# RC카와 object의 각도 측정 함수
def triAngle(p1, p2, p3):
    M_PI = 3.14159265358979323846
    numerator = p2[1]*(p1[0] - p3[0]) + p1[1]*(p3[0] - p2[0]) + p3[1]*(p2[0] - p1[0])
    denominator = (p2[0] - p1[0]) * (p1[0] - p3[0]) + (p2[1] - p1[1])*(p1[1] - p3[1])

    angleRad = math.atan2(numerator, denominator)
    angleDeg = int(angleRad * (180/M_PI))

    return angleDeg

# Rc카와 object의 거리 측정 함수
def objectDistance(d1, d2):
    
    distance = math.sqrt(pow(d1[0] - d2[0], 2) + pow(d1[1] - d2[1], 2))
    distance = int(distance)

    return distance

set_color = True
step = 0
# 색상 검출 함수
def color():
    global upper_RcFront, lower_RcFront
    global upper_RcCenter, lower_RcCenter
    global upper_Object1, lower_Object1
    global upper_Object2, lower_Object2
    global upper_Object3, lower_Object3
    global upper_EndPoint, lower_EndPoint
    # 빨강색 검출
    upper_RcFront = (9, 255, 227) 
    lower_RcFront = (0, 174, 55) 
    # 노랑색 검출
    upper_RcCenter= (40, 255, 255)
    lower_RcCenter = (19, 120, 102)   
    # 파랑색 
    upper_Object1 = (121, 255, 255)
    lower_Object1 = (97, 206, 102)    
    # 주황색
    upper_Object2 = (19, 255, 255)
    lower_Object2 = (7, 137, 102)
    # 초록색
    upper_Object3 = (97, 255, 255)
    lower_Object3 = (44, 81, 102)
    # 핑크색
    upper_EndPoint = (172, 146, 255)
    lower_EndPoint = (148, 42, 150)

# Rc카 이동제어를 위한 state
ObjectState1 = True
ObjectState2 = False
ObjectState3 = False 


# Rc카 및 object 좌표를 담을 리스트
ObjectPoint1 = []
ObjectPoint2 = []
ObjectPoint3 = []
RcCenterPoint = []
RcFrontPoint = []
EndPointList = []
# AngleList = []
# 아두이노 통신할 각도 변수
ardAngle = 0
fstObjectAngle = 0
sndObjectAngle = 0
trdObjectAngle = 0
EndPointAngle = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

color()
while(True):
    try:
        ret,img_color = cap.read()

        if ret == False:
            continue
        img_color2 = img_color.copy()
        img_hsv = cv2.cvtColor(img_color2, cv2.COLOR_BGR2HSV)

        height, width = img_color.shape[:2]
        cx = int(width / 2)
        cy = int(height / 2)

        if set_color == False:
            rectangle_color = (0, 255, 0)
            if step == 1:
                rectangle_color = (0, 0, 255)
            cv2.rectangle(img_color, (cx - 20, cy - 20), (cx + 20, cy + 20), rectangle_color, 5)
        else:
            img_RcFront = cv2.inRange(img_hsv, lower_RcFront, upper_RcFront)
            img_RcCenter = cv2.inRange(img_hsv, lower_RcCenter, upper_RcCenter)
            img_Object1 = cv2.inRange(img_hsv, lower_Object1, upper_Object1)
            img_Object2 = cv2.inRange(img_hsv, lower_Object2, upper_Object2)
            img_Object3 = cv2.inRange(img_hsv, lower_Object3, upper_Object3)
            img_EndPoint = cv2.inRange(img_hsv, lower_EndPoint, upper_EndPoint)
            # 모폴로지 연산
            kernel = np.ones((7,7), np.uint8)            
            img_RcFront = cv2.morphologyEx(img_RcFront, cv2.MORPH_OPEN, kernel)
            img_RcFront = cv2.morphologyEx(img_RcFront, cv2.MORPH_CLOSE, kernel)
          
            img_RcCenter = cv2.morphologyEx(img_RcCenter, cv2.MORPH_OPEN, kernel)
            img_RcCenter = cv2.morphologyEx(img_RcCenter, cv2.MORPH_CLOSE, kernel)

            img_Object1 = cv2.morphologyEx(img_Object1, cv2.MORPH_OPEN, kernel)
            img_Object1 = cv2.morphologyEx(img_Object1, cv2.MORPH_CLOSE, kernel)

            img_Object2 = cv2.morphologyEx(img_Object2, cv2.MORPH_OPEN, kernel)
            img_Object2 = cv2.morphologyEx(img_Object2, cv2.MORPH_CLOSE, kernel)

            img_Object3 = cv2.morphologyEx(img_Object3, cv2.MORPH_OPEN, kernel)
            img_Object3 = cv2.morphologyEx(img_Object3, cv2.MORPH_CLOSE, kernel)

            img_EndPoint = cv2.morphologyEx(img_EndPoint, cv2.MORPH_OPEN, kernel)
            img_EndPoint = cv2.morphologyEx(img_EndPoint, cv2.MORPH_CLOSE, kernel)

            # 라벨링
            # Rc카 중앙점 - 노랑
            numOfLabelsCenter, img_labelCenter, statsCenter, centroidsCenter = cv2.connectedComponentsWithStats(img_RcCenter)            
            for idx, centroid in enumerate(centroidsCenter):
                if statsCenter[idx][0] == 0 and statsCenter[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue
                x_Center,y_Center,width_Center,height_Center,area_Center = statsCenter[idx]
                centerX_Center,centerY_Center = int(centroid[0]), int(centroid[1])

                x1 = x_Center + int(width_Center/2)
                y1 = y_Center + int(height_Center/2)

                if len(RcCenterPoint) != 0:
                    RcCenterPoint.clear()
                    RcCenterPoint.extend([x1, y1])
                else: RcCenterPoint.extend([x1, y1])

                if area_Center > 100:
                    cv2.circle(img_color, (centerX_Center, centerY_Center), 4, (0,255,255), -1)
                    cv2.circle(img_color, (x_Center+width_Center, centerY_Center), 4, (0,255,255), -1)
                    cv2.circle(img_color, (centerX_Center, y_Center+height_Center), 4, (0,255,255), -1)
                    cv2.circle(img_color, (x_Center, centerY_Center), 4, (0,255,255), -1)
                    cv2.circle(img_color, (centerX_Center, y_Center), 4, (0,255,255), -1)
                    
                    text = "x: " + str(x1) + ", y: " + str(y1)
                    cv2.putText(img_color, text, (x1 + 10, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.rectangle(img_color, (x_Center,y_Center), (x_Center+width_Center,y_Center+height_Center), (0,255,255), thickness=3)
            
            # Rc카 앞쪽 포인트 - 빨강
            numOfLabelsFront, img_labelFront, statsFront, centroidsFront = cv2.connectedComponentsWithStats(img_RcFront)
           
            for idx, centroid in enumerate(centroidsFront):
                if statsFront[idx][0] == 0 and statsFront[idx][1] == 0:
                    continue

                if np.any(np.isnan(centroid)):
                    continue

                x_Front,y_Front,width_Front,height_Front,area_Front = statsFront[idx]
                centerX_Front,centerY_Front = int(centroid[0]), int(centroid[1])

                x2 = x_Front + int(width_Front/2)
                y2 = y_Front + int(height_Front/2)
                
                if len(RcFrontPoint) != 0:
                    RcFrontPoint.clear()
                    RcFrontPoint.extend([x2, y2])
                else: RcFrontPoint.extend([x2, y2])

                # 좌표 바운딩 상자
                if area_Front > 100:
                    cv2.circle(img_color, (centerX_Front, centerY_Front), 4, (0,0,255), -1)       
                    cv2.circle(img_color, (x_Front+width_Front, centerY_Front), 4, (0,0,255), -1)
                    cv2.circle(img_color, (x_Front, centerY_Front), 4, (0,0,255), -1)
                    cv2.circle(img_color, (centerX_Front, y_Front+height_Front), 4, (0,0,255), -1)
                    cv2.circle(img_color, (centerX_Front, y_Front), 4, (0,0,255), -1)
                    cv2.circle(img_color, (centerX_Front, y_Front), 4, (0,0,255), -1)
                    
                    text = "x: " + str(x2) + ", y: " + str(y2)
                    cv2.putText(img_color, text, (x2 + 10, y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.rectangle(img_color, (x_Front,y_Front), (x_Front+width_Front,y_Front+height_Front), (0,0,255), thickness=3)

            # 첫번째 오브젝트 -파랑
            numOfLabelsFst, img_labelFst, statsFst, centroidsFst = cv2.connectedComponentsWithStats(img_Object1)

            for idx, centroid in enumerate(centroidsFst):
                if statsFst[idx][0] == 0 and statsFst[idx][1] == 0:
                    continue

                if np.any(np.isnan(centroid)):
                    continue

                x_Fst,y_Fst,width_Fst,height_Fst,area_Fst = statsFst[idx]
                centerX_Fst,centerY_Fst = int(centroid[0]), int(centroid[1])
        
                x3 = x_Fst + int(width_Fst/2)
                y3 = y_Fst + int(height_Fst/2)

                if len(ObjectPoint1) != 0:
                    ObjectPoint1.clear()
                    ObjectPoint1.extend([x3, y3])
                else: ObjectPoint1.extend([x3, y3])

                # 좌표 바운딩 상자
                if area_Fst > 100:
                    cv2.circle(img_color, (centerX_Fst, centerY_Fst), 4, (255,0,0), -1)       
                    cv2.circle(img_color, (x_Fst+width_Fst, centerY_Fst), 4, (255,0,0), -1)
                    cv2.circle(img_color, (x_Fst, centerY_Fst), 4, (255,0,0), -1)
                    cv2.circle(img_color, (centerX_Fst, y_Fst+height_Fst), 4, (255,0,0), -1)
                    cv2.circle(img_color, (centerX_Fst, y_Fst), 4, (255,0,0), -1)
                    cv2.circle(img_color, (centerX_Fst, y_Fst), 4, (255,0,0), -1)
                    
                    text = "x: " + str(x3) + ", y: " + str(y3)
                    cv2.putText(img_color, text, (x3 + 10, y3 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.rectangle(img_color, (x_Fst,y_Fst), (x_Fst+width_Fst,y_Fst+height_Fst), (255,0,0), thickness=3)

            #두번째 오브젝트 - 주황
            numOfLabelsSnd, img_labelSnd, statsSnd, centroidsSnd = cv2.connectedComponentsWithStats(img_Object2)
             
            for idx, centroid in enumerate(centroidsSnd):
                if statsSnd[idx][0] == 0 and statsSnd[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue

                x_Snd,y_Snd,width_Snd,height_Snd,area_Snd = statsSnd[idx]
                centerX_Snd,centerY_Snd = int(centroid[0]), int(centroid[1])
                
                x4 = x_Snd + int(width_Snd/2)
                y4 = y_Snd + int(height_Snd/2)

                if len(ObjectPoint2) != 0:
                    ObjectPoint2.clear()
                    ObjectPoint2.extend([x4, y4])
                else: ObjectPoint2.extend([x4, y4])

                if area_Snd > 100:
                    cv2.circle(img_color, (centerX_Snd, centerY_Snd), 4, (128,128,255), -1)
                    cv2.circle(img_color, (x_Snd+width_Snd, centerY_Snd), 4, (128,128,255), -1)
                    cv2.circle(img_color, (centerX_Snd, y_Snd+height_Snd), 4, (128,128,255), -1)
                    cv2.circle(img_color, (x_Snd, centerY_Snd), 4, (128,128,255), -1)
                    cv2.circle(img_color, (centerX_Snd, y_Snd), 4, (128,128,255), -1)
                    
                    text = "x: " + str(x4) + ", y: " + str(y4)
                    cv2.putText(img_color, text, (x4 + 10, y4 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 2)
                    cv2.rectangle(img_color, (x_Snd,y_Snd), (x_Snd+width_Snd,y_Snd+height_Snd), (128,128,255),thickness=3)

            # 세번째 오브젝트 - 초록
            numOfLabelsTrd, img_labelTrd, statsTrd, centroidsTrd = cv2.connectedComponentsWithStats(img_Object3)
             
            for idx, centroid in enumerate(centroidsTrd):
                if statsTrd[idx][0] == 0 and statsTrd[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue

                x_Trd,y_Trd,width_Trd,height_Trd,area_Trd = statsTrd[idx]
                centerX_Trd,centerY_Trd = int(centroid[0]), int(centroid[1])
                            
                x5 = x_Trd + int(width_Trd/2)
                y5 = y_Trd + int(height_Trd/2)

                if len(ObjectPoint3) != 0:
                    ObjectPoint3.clear()
                    ObjectPoint3.extend([x5, y5])
                else: ObjectPoint3.extend([x5, y5])

                if area_Trd > 100:
                    cv2.circle(img_color, (centerX_Trd, centerY_Trd), 4, (0,255,0), -1)
                    cv2.circle(img_color, (x_Trd+width_Trd, centerY_Trd), 4, (0,255,0), -1)
                    cv2.circle(img_color, (centerX_Trd, y_Trd+height_Trd), 4, (0,255,0), -1)
                    cv2.circle(img_color, (x_Trd, centerY_Trd), 4, (0,255,0), -1)
                    cv2.circle(img_color, (centerX_Trd, y_Trd), 4, (0,255,0), -1)
                    
                    text = "x: " + str(x5) + ", y: " + str(y5)
                    cv2.putText(img_color, text, (x5 + 10, y5 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.rectangle(img_color, (x_Trd,y_Trd), (x_Trd+width_Trd,y_Trd+height_Trd), (0,255,0),thickness=3)
            
            
            # EndPoint - 핑크
            numOfLabelsEnd, img_labelEnd, statsEnd, centroidsEnd = cv2.connectedComponentsWithStats(img_EndPoint)
             
            for idx, centroid in enumerate(centroidsEnd):
                if statsEnd[idx][0] == 0 and statsEnd[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue

                x_End,y_End,width_End,height_End,area_End = statsEnd[idx]
                centerX_End,centerY_End = int(centroid[0]), int(centroid[1])
                
                x6 = x_End + int(width_End/2)
                y6 = y_End + int(height_End/2)

                if len(EndPointList) != 0:
                    EndPointList.clear()
                    EndPointList.extend([x6, y6])
                else: EndPointList.extend([x6, y6])

                if area_End > 100:
                    cv2.circle(img_color, (centerX_End, centerY_End), 4, (203,192,255), -1)
                    cv2.circle(img_color, (x_End+width_End, centerY_End), 4, (203,192,255), -1)
                    cv2.circle(img_color, (centerX_End, y_End+height_End), 4, (203,192,255), -1)
                    cv2.circle(img_color, (x_End, centerY_End), 4, (203,192,255), -1)
                    cv2.circle(img_color, (centerX_End, y_End), 4, (203,192,255), -1)
                    
                    text = "x: " + str(x6) + ", y: " + str(y6)
                    cv2.putText(img_color, text, (x6 + 10, y6 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (203,192,255), 2)
                    cv2.rectangle(img_color, (x_End,y_End), (x_End+width_End,y_End+height_End), (203,192,255),thickness=3)
            
            # 각도 확인.                  
            fstObjectAngle = triAngle(RcFrontPoint, RcCenterPoint, ObjectPoint1)
            sndObjectAngle = triAngle(RcFrontPoint, RcCenterPoint, ObjectPoint2)   
            trdObjectAngle = triAngle(RcFrontPoint, RcCenterPoint, ObjectPoint3)
            EndPointAngle = triAngle(RcFrontPoint, RcCenterPoint, EndPointList)
            
            distanceObject1 = objectDistance(ObjectPoint1, RcFrontPoint)         
            distanceObject2 = objectDistance(ObjectPoint2, RcFrontPoint)         
            distanceObject3 = objectDistance(ObjectPoint3, RcFrontPoint)
            distanceEnd = objectDistance(EndPointList, RcFrontPoint)

            limit = 110
            firstlimit = 160          
            if ObjectState1 == True:      
                if distanceObject1 > firstlimit:
                    if fstObjectAngle < -2:
                        ARD.write(b'R')
                        # print('first R')
                    elif fstObjectAngle > 2:
                        ARD.write(b'L')
                    elif fstObjectAngle <= 2 and fstObjectAngle >= -2:
                        ARD.write(b'F')
                elif distanceObject1 <= firstlimit:
                    ARD.write(b'S')
                    ObjectState1 = False
                    ObjectState2 = True
                    time.sleep(10)
                    
            if ObjectState2 == True:            
                if distanceObject2 > limit:
                    if sndObjectAngle < -2:
                        ARD.write(b'R')
                    elif sndObjectAngle > 2:
                        ARD.write(b'L')
                    elif sndObjectAngle <= 2 and sndObjectAngle >= -2:
                        ARD.write(b'F')
                elif distanceObject2 <= limit:
                    ARD.write(b'S')
                    ObjectState2 = False
                    ObjectState3 = True
                    time.sleep(10)
                
            if ObjectState3 == True:                
                if distanceObject3 > limit:
                    if trdObjectAngle < -2:
                        ARD.write(b'R')
                    elif trdObjectAngle > 2:
                        ARD.write(b'L')
                    elif trdObjectAngle <= 2 and trdObjectAngle >= -2:
                        ARD.write(b'F')
                elif distanceObject3 <= limit:
                    ARD.write(b'S')
                    time.sleep(10)
                    ObjectState3 = False
                    
            if ObjectState1 == False and ObjectState2 == False and ObjectState3 == False:
                if distanceEnd >= 50:
                    if EndPointAngle < -2:
                        ARD.write(b'R')
                    elif EndPointAngle > 2:
                        ARD.write(b'L')
                    elif EndPointAngle <= 2 and EndPointAngle >= -2:
                        ARD.write(b'F')
                elif distanceEnd < 50:
                    ARD.write(b'S')

        cv2.namedWindow('img_color', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('img_color', img_color)
             
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # esc
            break
    except IndexError:
        print("point is nothing")
        pass

cap.release()
cv2.destroyAllWindows()