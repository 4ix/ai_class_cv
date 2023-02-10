# Computer Vision
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