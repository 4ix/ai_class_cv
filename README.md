# Computer Vision
## 2023-03-03(금)
- Face Detector
1. [깃허브 링크](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

- Yolo Model
1. [DarkNet](https://pjreddie.com/darknet/yolo/)

## 2023-03-02(목)
- Deep Learning in OpenCV
1. [깃허브 링크](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV)
2. image classification(VGG16 활용)
3. image localization: 이미지 상 어디에 위치해 있는지 확인 (좌표를 찾아야 함)

- googlenet
1. [cafe model](http://dl.caffe.berkeleyvision.org/)
2. [config file](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt)
3. [분류 1000개](https://github.com/opencv/opencv/blob/4.x/samples/data/dnn/classification_classes_ILSVRC2012.txt)
4. [onix model](https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet)

## 2023-02-22(수)
- 영상의 히스토그램
```py
# 흑백
hist = cv2.calcHist([src], [0], None, [256], [0,256])

# 컬러
hist_b = cv2.calcHist([src], [0], None, [256], [0,256])
hist_g = cv2.calcHist([src], [1], None, [256], [0,256])
hist_r = cv2.calcHist([src], [2], None, [256], [0,256])
```

- 알파값 변경하면서 동영상으로 저장
```py
for i in range(100):
    alpha = i * 0.01
    dst = cv2.addWeighted(src, alpha, src_copy, 1-alpha, 0.)

    output.write(dst)

    cv2.imshow('dst', dst)
    if cv2.waitKey(100) == 27:
        break
    if i == 99:
        cv2.waitKey()
```

- 히스토그램 변환(흑백)
```py
# 흑백
src = cv2.imread('./fig/manjang.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
dst_norm = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, -1)
dst_equal = cv2.equalizeHist(src)
dst_hist = cv2.calcHist([dst_equal], [0], None, [256], [0, 256])

# 컬러
src = cv2.imread('./fig/autumn.jpg', cv2.IMREAD_REDUCED_COLOR_8)

# 1. equaliaziton
v_equal = cv2.equalizeHist(v)
dst_equal = cv2.merge((h, s, v_equal))
dst_equal = cv2.cvtColor(dst_equal, cv2.COLOR_HSV2BGR)

# 2. normaliazation
v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX, -1)
dst_norm = cv2.merge((h, s, v_norm))
dst_norm = cv2.cvtColor(dst_norm, cv2.COLOR_HSV2BGR)
```

- 특정 색상 영역 찾아내기
```py
src = cv2.imread('./fig/palette.png')
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100)) # green 을 뽑는다
dst2 = cv2.inRange(src_hsv, (40, 100, 0), (80, 255, 255)) # green 을 뽑는다
dst_hsv = cv2.inRange(src_hsv, (110, 100, 0), (130, 255, 255)) # blue 을 뽑는다
```

- 트랙바를 이용해서 색상찾기
```py
def call_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min', 'ctrl')
    hmax = cv2.getTrackbarPos('H_max', 'ctrl')
    smin = cv2.getTrackbarPos('S_min', 'ctrl')
    smax = cv2.getTrackbarPos('S_max', 'ctrl')
    vmin = cv2.getTrackbarPos('V_min', 'ctrl')
    vmax = cv2.getTrackbarPos('V_max', 'ctrl')

    dst = cv2.inRange(src_hsv, (hmin, smin, vmin), (hmax, smax, vmax))
    cv2.imshow('dst', dst)

cv2.createTrackbar('H_min', 'ctrl', 0, 179, call_trackbar)
cv2.createTrackbar('H_max', 'ctrl', 100, 179, call_trackbar)
...

```

## 2023-02-21(화)
- 트랙바
```py
def call_trackbar(pos):
    src[:] = (img/255) * pos
    cv2.imshow('src', src)

cv2.imshow('src', src)
cv2.createTrackbar('level', 'src', 0, 255, call_trackbar)
```

- 시간 체크
```py
tm.start()
for _ in range(100):
    img = cv2.GaussianBlur(src, (0,0), 5)

tm.stop()
t2 = time.time()
print(tm.getTimeMilli(), 'ms')

```

- 산술연산
```py
dst = cv2.add(src, (100, 100, 100, 0))

dst1 = cv2.add(src1, src2)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)
```

- 비트연산
```py
bit_and = cv2.bitwise_and(src1, src2)
bit_or = cv2.bitwise_or(src1, src2)
bit_xor = cv2.bitwise_xor(src1, src2)
bit_not = cv2.bitwise_not(src2)

```

- 컬러영상
```py
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(src_hsv) # hue 값은 최대값이 179 정도라서 밝게 나올 수가 없음
v_1 = cv2.add(v, -50) # 색상은 보존하면서 밝기만 조절하고 싶을 때

src_merge = cv2.merge((h, s, v_1))
src_merge = cv2.cvtColor(src_merge, cv2.COLOR_HSV2BGR) # BGR로 바꿔줘야 밝기 조절이 가능함
```


## 2023-02-17(금)
- 동영상
```py
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('open failed')
    sys.exit()

f_flag = False
i_flag = False

while True:
    ret, frame = cap.read()

    if not ret:
        print('frame read failed')
        break
    if f_flag == True:
        frame = cv2.flip(frame, 1)
    if i_flag == True:
        frame = cv2.Canny(frame, 50, 150)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(20)
    
    if key == 27:
        break
    elif key == ord('f'):
        f_flag = not f_flag
    elif key == ord('i'):
        i_flag = not i_flag

    
cap.release()
cv2.destroyAllWindows()
```

- 마우스 콜백
```py
def call_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'left button clicked = {x}, {y}')
    elif event == cv2.EVENT_LBUTTONUP:
        print(f'left button up = {x}, {y}')
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            print(f'moving = {x}, {y}')

img = np.ones((480, 640, 3), np.uint8) * 255

cv2.namedWindow('img')
cv2.setMouseCallback('img', call_mouse, img)

cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 2023-02-16(목)
- 이미지 필터
```py
current = cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE)
...
current = cv2.Canny(current, 50, 150)
...
current = cv2.GaussianBlur(current, (0, 0), 1)
...
```


## 2023-02-15(수)
- 알파 채널 이용하기
```py
src = cv2.imread('./fig/ch2_fig/imgbin_hat.png', cv2.IMREAD_UNCHANGED)
src = cv2.resize(src,(250, 180))

src_img = src[:,:,:-1] # 알파채널 빼고 읽기
src_mask = src[:,:,-1] # 알파채널만 읽기

```

## 2023-02-14(화)
- 마스크 만들기
```py
img1 = cv2.imread('./fig/ch2_fig/cow.png')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변경

_, img_mask = cv2.threshold(img1_gray, 240, 250, cv2.THRESH_BINARY_INV) # 그레이 이미지를 넣고, 240부터 250까지를 1로 만든다는 소리임
```

- 서로 다른 이미지 합치기
```py
cv2.copyTo(src, mask, dst) # 이미지 크기가 같아야 하고, mask 영상이 존재해야함
```

[서로다른 이미지 합치기](https://yeko90.tistory.com/entry/opencv-%EB%91%90-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%95%A9%EC%B9%98%EB%8A%94-%EB%B0%A9%EB%B2%95-%ED%81%AC%EA%B8%B0-%EB%8B%A4%EB%A5%B8-%EC%9D%B4%EB%AF%B8%EC%A7%80)


## 2023-02-10(금)
- 기본적인 이미지 호출 방법
```py
img = cv2.imread('./fig/puppy.bmp', cv2.IMREAD_REDUCED_COLOR_2)

print(type(img))
print(img.shape)

if img is None:
    print('image read failed')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)

k = cv2.waitKey()
print(k)

cv2.destroyAllWindows()
```

- cv2로 읽어들이는 이미지는 가로, 세로, RGB 순서임
- numpy는 반대임(세로, 가로, BGR)