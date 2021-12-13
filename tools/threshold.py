import cv2
import numpy as np

if __name__ == '__main__':
    image_name = '/home/li-qiufu/Downloads/1705875717.jpg'
    image = cv2.imread(image_name, 0)
    print(image.shape)
    print(image[100,:])
    image_new = image
    image_new[image_new > 150] = 255
    image_new[image_new <= 150] = 20
    cv2.imwrite('/home/li-qiufu/Downloads/1705875717_new.jpg', image_new)
    cv2.imshow('image', image)
    cv2.imshow('image_new', image_new)
    cv2.waitKey(0)