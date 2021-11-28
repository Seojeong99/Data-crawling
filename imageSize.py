import cv2

img = cv2.imread('D:/testImage/2021-11-19/14/2021-11-19-14-46-49.jpg')

h, w, c = img.shape
print('width:  ', w)
print('height: ', h)