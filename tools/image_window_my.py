import cv2
import os
import numpy as np
#import
#from pygame.locals import *

SUFFIX = ['.bmp', '.tiff', '.jpeg', '.jpg', '.png']

class ImageWindow():
    """
    这个类定义一个图像显示窗口，在这个窗口下显示某个文件夹的一系列图片，
    这个窗口接受键盘输入，按上下键或左右键在窗口中依次对图片进行翻页
    """
    def __init__(self, image_root, cache = 10):
        self.image_root = image_root
        self.cache = cache
        self.image_name_list = os.listdir(self.image_root)
        self.image_name_list.sort()
        cv2.namedWindow(self.image_root, cv2.WINDOW_FREERATIO)
        self.number = 0
        self.numbers_cache = []
        self.images_cache = []
        self.get_image_data()

    def get_image_name(self):
        while True:
            assert len(self.image_name_list) > 0, '指定的文件夹中可能没有图片数据文件 ...'
            image_name = self.image_name_list[self.number]
            _, suffix = os.path.splitext(image_name)
            if suffix not in SUFFIX:
                self.image_name_list.pop(self.number)
            else:
                if len(self.numbers_cache) >= self.cache:
                    self.numbers_cache.pop(0)
                self.numbers_cache.append(self.number)
                return image_name

    def get_image_data(self):
        if self.number in self.numbers_cache:
            return self.images_cache[self.numbers_cache.index(self.number)]
        image_name = self.get_image_name()
        image_full_name = os.path.join(self.image_root, image_name)
        image = cv2.imread(image_full_name, 1)
        if len(self.images_cache) >= self.cache:
            self.images_cache.pop(0)
        self.images_cache.append(image)
        return image

    def show(self):
        image = self.get_image_data()
        cv2.imshow(self.image_root, image)
        key_word = cv2.waitKey(0)
        while True:
            if key_word == 81 or key_word == 82:    # 收到反向键的“左”或“上”，往前翻一张
                if self.number == 0:
                    pass
                else:
                    self.number -= 1
                    image = self.get_image_data()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = self.image_name_list[self.number]
                    cv2.putText(image, text, org = (30, 30), fontFace = font, fontScale = 1.2, color = (255, 0, 0), thickness=1)
                    cv2.imshow(self.image_root, image)
            elif key_word == 83 or key_word == 84:    # 收到反向键的“右”或“下”，往后翻一张
                if self.number + 1 == len(self.image_name_list):
                    pass
                else:
                    self.number += 1
                    image = self.get_image_data()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = self.image_name_list[self.number]
                    cv2.putText(image, text, org = (30, 30), fontFace = font, fontScale = 1.2, color = (255, 0, 0), thickness=1)
                    cv2.imshow(self.image_root, image)
            elif key_word == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                continue
            key_word = cv2.waitKey(0)


if __name__ == '__main__':
    image_root_save = '/home/li-qiufu/PycharmProjects/MyDataBase/DataBase_8_rotate/000074/image'
    #image = cv2.imread(image_root_save,0)
    #cv2.imshow('image',image)
    #while True:
    #    key = cv2.waitKey(0)
    #    print(key)
    #    if key == ord('q'):
    #        break
    ImageWindow(image_root_save).show()