# 在生成 main 文件时, 请勾选该模块

# 导入必要的包
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

def spilt_data(nPerson, nPicture, data, label):
    """
    分割数据集

    :param nPerson : 志愿者数量
    :param nPicture: 各志愿者选入训练集的照片数量
    :param data : 等待分割的数据集
    :param label: 对应数据集的标签
    :return: 训练集, 训练集标签, 测试集, 测试集标签
    """
    # 数据集大小和意义
    allPerson, allPicture, rows, cols = data.shape

    # 划分训练集和测试集
    train = data[:nPerson,:nPicture,:,:].reshape(nPerson*nPicture, rows*cols)
    train_label = label[:nPerson, :nPicture].reshape(nPerson * nPicture)
    test = data[:nPerson, nPicture:, :, :].reshape(nPerson*(allPicture - nPicture), rows*cols)
    test_label = label[:nPerson, nPicture:].reshape(nPerson * (allPicture - nPicture))

    # 返回: 训练集, 训练集标签, 测试集, 测试集标签
    return train, train_label, test, test_label

datapath = './ORL.npz'
ORL = np.load(datapath)
data = ORL['data']
label = ORL['label']
num_eigenface = 200

train_vectors, train_labels, test_vectors, test_labels = spilt_data(40, 5, data,label)
train_vectors = train_vectors / 255
test_vectors = test_vectors / 255

print("训练数据集:", train_vectors.shape)
print("测试数据集:", test_vectors.shape)


def plot_gallery(images, titles, n_row=3, n_col=5, h=112, w=92):  # 3行4列
    """
    展示多张图片

    :param images: numpy array 格式的图片
    :param titles: 图片标题
    :param h: 图像reshape的高
    :param w: 图像reshape的宽
    :param n_row: 展示行数
    :param n_col: 展示列数
    :return:
    """
    # 展示图片
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def eigen_train(trainset, k=20):
    """
    训练特征脸（eigenface）算法的实现

    :param trainset: 使用 get_images 函数得到的处理好的人脸数据训练集
    :param K: 希望提取的主特征数
    :return: 训练数据的平均脸, 特征脸向量, 中心化训练数据
    """
    print("步骤1/5: 计算平均人脸...")
    avg_img = np.mean(trainset, axis=0)
    
    print("步骤2/5: 计算中心化人脸...")
    norm_img = trainset - avg_img
    
    print("步骤3-4/5: 使用SVD直接计算特征值和特征向量...")
    # 使用SVD避免计算大型协方差矩阵
    U, S, Vh = np.linalg.svd(norm_img, full_matrices=False)
    eigen_values = S**2 / (len(norm_img) - 1)  # 转换为特征值
    eigen_vectors = Vh.T  # Vh的转置就是我们需要的特征向量
    
    # 排序（SVD通常已经按降序排列，但为安全起见）
    idx = np.argsort(-eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    
    print("步骤5/5: 选择前%d个特征向量..." % k)
    feature = eigen_vectors[:, 0:min(k, eigen_vectors.shape[1])].T
    
    print("训练完成!")
    return avg_img, feature, norm_img
    
avg_img, eigenface_vects, trainset_vects = eigen_train(train_vectors, num_eigenface)

# 在生成 main 文件时, 请勾选该模块

def rep_face(image, avg_img, eigenface_vects, numComponents = 0):
    """
    用特征脸（eigenface）算法对输入数据进行投影映射，得到使用特征脸向量表示的数据

    :param image: 输入数据
    :param avg_img: 训练集的平均人脸数据
    :param eigenface_vects: 特征脸向量
    :param numComponents: 选用的特征脸数量
    :return: 输入数据的特征向量表示, 最终使用的特征脸数量
    """
    numEigenFaces=min(numComponents,eigenface_vects.shape[0])
    w=eigenface_vects[0:numEigenFaces,].T
    representation=np.dot(image-avg_img,w)
    ###################################################################################
    ####  用特征脸（eigenface）算法对输入数据进行投影映射，得到使用特征脸向量表示的数据  ####
    ####                          请勿修改该函数的输入输出                           ####
    ###################################################################################
    #                                                                                 #

    #                                                                                 #
    ###################################################################################
    #############             在生成 main 文件时, 请勾选该模块              #############
    ###################################################################################

    # 返回：输入数据的特征向量表示, 特征脸使用数量
    return representation, numEigenFaces

train_reps = []
for img in train_vectors:
    train_rep, _ = rep_face(img, avg_img, eigenface_vects, num_eigenface)
    train_reps.append(train_rep)

num = 0
for idx, image in enumerate(test_vectors):
    label = test_labels[idx]
    test_rep, _ = rep_face(image, avg_img, eigenface_vects, num_eigenface)

    results = []
    for train_rep in train_reps:
        similarity = np.sum(np.square(train_rep - test_rep))
        results.append(similarity)
    results = np.array(results)

    if label == np.argmin(results) // 5 + 1:
        num = num + 1

print("人脸识别准确率: {}%".format(num / 80 * 100))
