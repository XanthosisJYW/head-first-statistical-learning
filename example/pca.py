import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 0. 辅助函数 - 绘制箭头
def draw_vector(start_point, end_point, ax=None):
    ax = ax or plt.gca()
    arrow_props = dict(arrowstyle='->',
                       linewidth=2,
                       shrinkA=0,
                       shrinkB=0)
    ax.annotate('', end_point, start_point, arrowprops=arrow_props)


# 1. 生成模拟数据
random_generator = np.random.RandomState(1)
data = np.dot(random_generator.rand(2, 2), random_generator.randn(2, 200)).T

# 2. 进行主成分分析
pca = PCA(n_components=2)
pca.fit(data)
print(pca.components_)

# 3. 绘制原始图像
figure_one = plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], marker="o", color="blue", alpha=0.3)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
figure_one.show()

# 4. 绘制旋转后的图像
figure_two = plt.figure(2)
data_rotated = np.dot(data, pca.components_)
plt.scatter(data_rotated[:, 0], data_rotated[:, 1], marker="o", color="red", alpha=0.3)
plt.axis('equal')
figure_two.show()

plt.show()
