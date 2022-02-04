<h2 align="center">
  dlib face detection
</h2>

<div align="center">
  <img src="https://img.shields.io/badge/python-v3.10-blue.svg"/>
  <img src="https://img.shields.io/badge/dlib-v19.23.0-blue.svg"/>
  <img src="https://img.shields.io/badge/face_recognition-v1.3.0-blue.svg"/>
</div>

스마트폰의 카메라를 사용해 보셨다면 한번쯤은 얼굴 인식 기능을 경험해 보셨을 겁니다. 최근 카메라의 얼굴 인식 기능은 매우 중요한 기능이 되었습니다. 얼굴을 자동으로 인식하고 초점을 맞추거나 태그를 만들어 주기도 합니다.

<div align="center">
  <a href="https://yunwoong.tistory.com/83" target="_blank" title="dlib, Python을 이용하여 얼굴 검출하기" rel="nofollow">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmQIvq%2FbtrrfmJY8xk%2FA9zejVsPOXWDMGpXFJRAA1%2Fimg.png" width="500" title="dlib face detection" alt="dlib face detection">
    </img>
  </a>
</div>

얼굴 인식 기술은 여러 가지 모델이 제안되었는데 OpenCV의 Harr Cascades와 dlib의 HOG (Histogram of Oriented Gradients) 가 대표적인 모델입니다. 여기서는 **dlib의 HOG** 방식을 사용 할 것입니다.

Python과 dlib을 이용하여 간단하게 **얼굴 검출(face detection) 기능**을 구현하는 방법을 소개하도록 하겠습니다. 직접적으로 dlib을 사용해도 되지만 여기서는 Python의 face recognition 라이브러리를 이용하도록 하겠습니다. face recognition은 간단한 얼굴 인식 라이브러리로 dlib기반으로 구축되었습니다. face recognition에서 사용되는 모델은 [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) 기준으로 99.38%의 정확도를 가진다고 소개하고 있네요.

우선, dlib이 이미 설치가 되어 있어야 합니다. 만약 설치되어 있지 않다면 [dlib 설치가이드](https://yunwoong.tistory.com/80)를 참고하시여 설치를 진행하시기 바랍니다.

#### **1. Install**

```python
pip install face_recognition
```

#### **2. Import Packages**

```python
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import dlib
import cv2
 
import face_recognition
```

#### **3. Function**

Colab 또는 Jupyter Notebook에서 이미지를 확인하기 위한 Function입니다.

```python
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

#### **4. \**Declare a face detector\****

```python
# face_landmark_path = 'lib/landmark/shape_predictor_68_face_landmarks.dat'
# predictor = dlib.shape_predictor(face_landmark_path)
detector = dlib.get_frontal_face_detector()
predictor = face_recognition.api.pose_predictor_68_point
```

#### **5. Load Image**

```python
image_path = 'asset/images/2021_g7.jpg' 
org_image = cv2.imread(image_path) 
image = org_image.copy() 
image = imutils.resize(image, width=500) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
 
rects = detector(gray, 1)
```

#### **6. Face detection**

```python
for (i, rect) in enumerate(rects):
    # 얼굴 영역의 얼굴 랜드마크를 결정한 다음 
    # 얼굴 랜드마크(x, y) 좌표를 NumPy Array로 변환합니다.
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # dlib의 사각형을 OpenCV bounding box로 변환(x, y, w, h)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 얼굴 랜드마크에 포인트를 그립니다.
    for (i, (x, y)) in enumerate(shape):
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        # cv2.putText(image, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
        
plt_imshow("Output", image, figsize=(16,10))
```

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbc4b0n%2FbtrrjhAT1Xx%2FVLNIomjhEZQTTgWpoka7Vk%2Fimg.png" width="100%">
</div>

얼굴 검출은 매우 잘되는 것 같습니다. 다만 옆모습이거나 얼굴이 일부 가려진 경우에는 인식을 못하는 경우도 있습니다.

------

갑자기 여기 이 사진 속 사람들 몇명일까..? 하는 생각이 들어서 수행해 보았습니다. (와..많네요..@.@)

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQEafR%2FbtrrfmJ6o8U%2FM6Hi3zIoKAgRVUVdiCps91%2Fimg.png" width="70%">
</div>

수행 결과 입니다. 112명...이라고 나오네요. 물론 고개를 숙이고 있거나 얼굴이 가려진 사람들은 인식을 하지 못했습니다. 하지만 이 결과를 보면서 사람이 하는일을 편하게 하거나 손쉽게 하는 방법이 있을거라는 생각은 드네요.

<div align="center">
  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTMLRF%2FbtrrdXX2yJa%2F8zGhQmCKd6F8dEhTd1l48K%2Fimg.png" width="70%">
</div>

------

**조금 더 나아가서..**

위에 결과를 확인 할 수 있듯이 옆모습이거나 얼굴이 가려진 경우에는 인식이 안된다는 걸 알 수 있습니다. 다음 글에는 dlib보다 좀 더 방법을 소개 하도록 하겠습니다.

<div align="center">
  <img src="/asset/images/img.gif" width="50%">
</div>
