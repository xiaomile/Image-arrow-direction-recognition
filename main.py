import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def conv(image, kernel, mode='same'):
    
 #进行卷积运算
    res = _convolve(image[:, :], kernel)
    return res

def normal(image, kernel):
#np.multiply()函数：数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致（点对点相乘）
    res = np.multiply(image, kernel).sum()
    if res > 255:
        return 255
    elif res<0:
        return 0
    else:
        return res

def _convolve(image, kernel):
    h_kernel, w_kernel = kernel.shape#获取卷积核的长宽，也就是行数和列数
    h_image, w_image = image.shape#获取欲处理图片的长宽
 #计算卷积核中心点开始运动的点，因为图片边缘不能为空
    res_h = h_image - h_kernel + 1
    res_w = w_image - w_kernel + 1
#生成一个0矩阵，用于保存处理后的图片
    res = np.zeros((res_h, res_w), np.uint8)
    for i in range(res_h):
        for j in range(res_w):
#image处传入的是一个与卷积核一样大小矩阵，这个矩阵取自于欲处理图片的一部分
            #这个矩阵与卷核进行运算，用i与j来进行卷积核滑动
            res[i, j] = normal(image[i:i + h_kernel, j:j + w_kernel], kernel)

    return res

kernel_r = np.array([[0,0,1,1,1,0,0,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,0,0,1,3,1,0,0],
                     [1,1,1,1,1,1,3,1,0],
                     [1,3,3,3,3,3,3,3,1],
                     [1,1,1,1,1,1,3,1,0],
                     [0,0,0,0,1,3,1,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,1,1,1,0,0,0,0],])

kernel_u = np.array([[0,0,0,0,1,0,0,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,1,3,3,3,1,0,0],
                     [0,1,3,1,3,1,3,1,0],
                     [1,3,1,1,3,1,1,3,1],
                     [1,1,0,1,3,1,0,1,1],
                     [1,0,0,1,3,1,0,0,1],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,0,1,1,1,0,0,0],])

kernel_l = np.array([[0,0,0,0,1,1,1,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,1,3,1,0,0,0,0],
                     [0,1,3,1,1,1,1,1,1],
                     [1,3,3,3,3,3,3,3,1],
                     [0,1,3,1,1,1,1,1,1],
                     [0,0,1,3,1,0,0,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,0,0,1,1,1,0,0],])

kernel_d = np.array([[0,0,0,1,1,1,0,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [1,0,0,1,3,1,0,0,1],
                     [1,1,0,1,3,1,0,1,1],
                     [1,3,1,1,3,1,1,3,1],
                     [0,1,3,1,3,1,3,1,0],
                     [0,0,1,3,3,3,1,0,0],
                     [0,0,0,1,3,1,0,0,0],
                     [0,0,0,0,1,0,0,0,0],])

if __name__ == '__main__':
    imgdir = 'images'
    imgpaths = [imgdir+'/'+filename for filename in os.listdir(imgdir) if '.jpg' in filename ]
    
    plt.figure()
    n=0
    for imgpath in imgpaths:
        n+=1
        img = cv2.imread(imgpath)
        img_w,img_h = img.shape[:2]
        if img_w<img_h:
            diff = int((img_h-img_w)/2)
            img2 = img[:,diff:img_h-diff,:]
        else:
            diff = img_w-img_h
            img2 = img[diff:img_w-diff,:,:]
        img4 = cv2.resize(img2,(224,224))
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img5=cv2.resize(img3,(27,27))
        ret,img6 = cv2.threshold(img5, 127, 255, cv2.THRESH_BINARY)
        #print(img6)
        #cv2.imshow('',img6)
        img7 = img6/255
        result_list = []
        kernel_dict = {'right':kernel_r,'up':kernel_u,'left':kernel_l,'down':kernel_d}
        for kernel in kernel_dict.keys():
            #print(conv(img7, kernel_dict[kernel]))
            result_list.append({kernel:np.max(conv(img7, kernel_dict[kernel]))})
        result_list.sort(key=lambda x:max(x.values()),reverse=True)
        #print(imgpath,''.join(result_list[0].keys()))
        plt.subplot(4,4,n)
        plt.axis('off')
        plt.imshow(img4)
        plt.title(imgpath.replace(imgdir+'/','')+'('+''.join(result_list[0].keys())+')')
    plt.show()
    
    
