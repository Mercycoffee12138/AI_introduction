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

# 在生成 main 文件时, 请勾选该模块

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
# 返回平均人脸、特征人脸、中心化人脸
avg_img, eigenface_vects, trainset_vects = eigen_train(train_vectors, num_eigenface)

# 打印两张特征人脸作为展示
eigenfaces = eigenface_vects.reshape((num_eigenface, 112, 92))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, n_row=1, n_col=2)
