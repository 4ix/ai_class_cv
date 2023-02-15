# Computer Vision
## 2023-02-15(수)
- 알파 채널 이용하기
```
src = cv2.imread('./fig/ch2_fig/imgbin_hat.png', cv2.IMREAD_UNCHANGED)
src = cv2.resize(src,(250, 180))

src_img = src[:,:,:-1] # 알파채널 빼고 읽기
src_mask = src[:,:,-1] # 알파채널만 읽기

```

## 2023-02-14(화)
- 마스크 만들기
```
img1 = cv2.imread('./fig/ch2_fig/cow.png')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변경

_, img_mask = cv2.threshold(img1_gray, 240, 250, cv2.THRESH_BINARY_INV) # 그레이 이미지를 넣고, 240부터 250까지를 1로 만든다는 소리임
```

- 서로 다른 이미지 합치기
```
cv2.copyTo(src, mask, dst) # 이미지 크기가 같아야 하고, mask 영상이 존재해야함
```

[서로다른 이미지 합치기](https://yeko90.tistory.com/entry/opencv-%EB%91%90-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%95%A9%EC%B9%98%EB%8A%94-%EB%B0%A9%EB%B2%95-%ED%81%AC%EA%B8%B0-%EB%8B%A4%EB%A5%B8-%EC%9D%B4%EB%AF%B8%EC%A7%80)


## 2023-02-10(금)
- 기본적인 이미지 호출 방법
```
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