from utils_de import *  # 导入自定义的工具包，包含自定义的函数，如 imread、imwrite、preprocess 等
import glob  # 用于查找符合特定规则的文件路径名
import os  # 提供了一些与操作系统进行交互的函数
import matplotlib.pyplot as plt  # 用于数据可视化的库
import numpy as np  # 用于数组和矩阵操作的科学计算库
from multiprocessing.pool import Pool  # 多进程处理模块，用于并行化处理
import cv2 as cv  # OpenCV库，用于图像处理
import csv  # 用于CSV文件的读写操作
from PIL import ImageFile  # 用于处理图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 设置为True可以加载部分图像（即使图像被截断）

dsize = (512, 512)  # 定义图像大小为512x512

def process(image_list):  # 定义处理图像的函数
    for image_path in image_list:  # 遍历图像路径列表
        name = image_path.split('/')[-1]  # 获取图像文件名
        dst_image_path = os.path.join('./data/image', name)  # 定义处理后的图像保存路径
        dst_mask_path = os.path.join('./data/mask', name)  # 定义处理后的掩码图像保存路径
        try:
            img = imread(image_path)  # 读取图像
            img, mask = preprocess(img)  # 预处理图像，返回图像和对应的掩码
            # img = cv.resize(img, dsize)  # 调整图像大小
            # mask = cv.resize(mask, dsize)  # 调整掩码大小
            imwrite(dst_image_path, img)  # 保存处理后的图像
            imwrite(dst_mask_path, mask)  # 保存处理后的掩码
        except:
            print(image_path)  # 如果处理过程中出现错误，打印出错的图像路径
            continue  # 跳过当前图像，继续处理下一个图像

if __name__=="__main__":  # 主程序入口
    image_list = glob.glob(os.path.join('./data/sample', '*.png'))  # 获取所有JPEG格式的图像路径列表
    patches = 16  # 定义要分割的子任务数量
    patch_len = int(len(image_list) / patches)  # 计算每个子任务中包含的图像数量
    filesPatchList = []  # 创建一个空列表，用于存储子任务的图像列表
    for i in range(patches - 1):  # 遍历除最后一个外的所有子任务
        fileList = image_list[i * patch_len:(i + 1) * patch_len]  # 获取每个子任务的图像列表
        filesPatchList.append(fileList)  # 将图像列表添加到子任务列表中
    filesPatchList.append(image_list[(patches - 1) * patch_len:])  # 将最后一个子任务的图像列表添加到子任务列表中

    # 多进程处理
    pool = Pool(patches)  # 创建一个包含指定数量（patches）的进程池
    pool.map(process, filesPatchList)  # 并行地将每个子任务分配给进程池中的进程
    pool.close()  # 关闭进程池，不再接受新的任务
