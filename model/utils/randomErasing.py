import math
import random

class RandomErasing(object):
    """ Randomly selects a rectangle region in a image and erases its pixels.
       'Random Erasing Data Augmentation' by Zhong et al.
       See https://arxiv.org/pdf/1708.04896.pdf
     Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.最大擦除面积
         sh: Maximum propertion of erased area against input image.最小擦除面积
         r1: Minimum aspect ratio of erased area.擦除面积的最小纵横比
         mean: Erasing value.擦除填充值
     """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        
    def __call__(self, img):
        if random.uniform(0,1) >= self.probability:
            return img
        for attempt in range(100):
            # img.size()[1] * img.size()[2] = h*w 
            area = img.size()[1] * img.size()[2]
            
            target_area = random.uniform(self.sl, self.self.sh) * area #擦除比例
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h) #随机选取一个x
                y1 = random.randint(0, img.size()[2] - w) #随机选取一个y
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w]
                return img
        return img
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            