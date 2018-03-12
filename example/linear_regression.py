import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 读取数据
data_all = pd.read_csv("../data/Advertising.csv", index_col=0)
print(data_all.head())
print("\n")

# 2.1 电视 & 销量关系图
data_tv_sales = data_all[["TV", "Sales"]]
model_tv_sales = smf.ols(formula="Sales ~ TV", data=data_tv_sales)
result_tv_sales = model_tv_sales.fit()
y_tv_predicted = result_tv_sales.predict(data_all["TV"])  # 预测 (回归)

figure_one = plt.figure(1)
plt.scatter(data_all["TV"], data_all["Sales"], marker="o", color="blue", label="Original")
plt.plot(data_all["TV"], y_tv_predicted, color="red", label="Predicted")
plt.legend()
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Sales - TV")

figure_one.show()
print(result_tv_sales.summary())
print("\n")

# 2.2 广播 & 销量关系图
data_radio_sales = data_all[["Radio", "Sales"]]
model_radio_sales = smf.ols(formula="Sales ~ Radio", data=data_radio_sales)
result_radio_sales = model_radio_sales.fit()
y_radio_predicted = result_radio_sales.predict(data_all["Radio"])  # 预测 (回归)

figure_two = plt.figure(2)
plt.scatter(data_all["Radio"], data_all["Sales"], marker="o", color="green", label="Original")
plt.plot(data_all["Radio"], y_radio_predicted, color="red", label="Predicted")
plt.legend()
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.title("Sales - Radio")

figure_two.show()
print(result_radio_sales.summary())
print("\n")

# 2.3 报纸 & 销量关系图
data_newspaper_sales = data_all[["Newspaper", "Sales"]]
model_newspaper_sales = smf.ols(formula="Sales ~ Newspaper", data=data_newspaper_sales)
result_newspaper_sales = model_newspaper_sales.fit()
y_newspaper_predicted = result_newspaper_sales.predict(data_all["Newspaper"])  # 预测 (回归)

figure_three = plt.figure(3)
plt.scatter(data_all["Newspaper"], data_all["Sales"], marker="o", color="yellow", label="Original")
plt.plot(data_all["Newspaper"], y_newspaper_predicted, color="red", label="Predicted")
plt.legend()
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.title("Sales - Newspaper")

figure_three.show()
print(result_newspaper_sales.summary())
print("\n")

# 3.1 电视, 广播 & 销量关系图
data_tv_radio_sales = data_all[["TV", "Radio", "Sales"]]
model_tv_radio_sales = smf.ols(formula="Sales ~ TV + Radio", data=data_tv_radio_sales)
result_tv_radio_sales = model_tv_radio_sales.fit()

x_tv, y_radio = np.meshgrid(np.linspace(data_all["TV"].min(), data_all["TV"].max(), 100),
                            np.linspace(data_all["Radio"].min(), data_all["Radio"].max(), 100))
xy_tv_radio = pd.DataFrame({"TV": x_tv.ravel(), "Radio": y_radio.ravel()})
z_tv_radio_predicted = result_tv_radio_sales.predict(xy_tv_radio)
axes_four = plt.figure(4).add_subplot(111, projection='3d')
axes_four.scatter(data_all["TV"], data_all["Radio"], data_all["Sales"], marker="o", color="blue")
axes_four.plot_surface(x_tv, y_radio, z_tv_radio_predicted.values.reshape(x_tv.shape), color="red", alpha=0.2)
axes_four.set_xlabel("TV")
axes_four.set_ylabel("Radio")
axes_four.set_zlabel("Sales")
axes_four.set_title("Sales - TV + Radio")

print(result_tv_radio_sales.summary())
print("\n")

# 3.2 电视, 报纸 & 销量关系图
data_tv_newspaper_sales = data_all[["TV", "Newspaper", "Sales"]]
model_tv_newspaper_sales = smf.ols(formula="Sales ~ TV + Newspaper", data=data_tv_newspaper_sales)
result_tv_newspaper_sales = model_tv_newspaper_sales.fit()

x_tv, y_newspaper = np.meshgrid(np.linspace(data_all["TV"].min(), data_all["TV"].max(), 100),
                                np.linspace(data_all["Newspaper"].min(), data_all["Newspaper"].max(),
                                            100))
xy_tv_newspaper = pd.DataFrame({"TV": x_tv.ravel(), "Newspaper": y_newspaper.ravel()})
z_tv_newspaper_predicted = result_tv_newspaper_sales.predict(xy_tv_newspaper)
axes_five = plt.figure(5).add_subplot(111, projection='3d')
axes_five.scatter(data_all["TV"], data_all["Newspaper"], data_all["Sales"], marker="o", color="blue")
axes_five.plot_surface(x_tv, y_newspaper, z_tv_newspaper_predicted.values.reshape(x_tv.shape),
                       color="red", alpha=0.2)
axes_five.set_xlabel("TV")
axes_five.set_ylabel("Newspaper")
axes_five.set_zlabel("Sales")
axes_five.set_title("Sales - TV + Newspaper")

print(result_tv_newspaper_sales.summary())
print("\n")

# 3.3 广播, 报纸 & 销量关系图
data_radio_newspaper_sales = data_all[["Radio", "Newspaper", "Sales"]]
model_radio_newspaper_sales = smf.ols(formula="Sales ~ Radio + Newspaper", data=data_radio_newspaper_sales)
result_radio_newspaper_sales = model_radio_newspaper_sales.fit()

x_radio, y_newspaper = np.meshgrid(np.linspace(data_all["Radio"].min(), data_all["Radio"].max(), 100),
                                   np.linspace(data_all["Newspaper"].min(), data_all["Newspaper"].max(),
                                               100))
xy_radio_newspaper = pd.DataFrame({"Radio": x_radio.ravel(), "Newspaper": y_newspaper.ravel()})
z_radio_newspaper_predicted = result_radio_newspaper_sales.predict(xy_radio_newspaper)
axes_six = plt.figure(6).add_subplot(111, projection='3d')
axes_six.scatter(data_all["Radio"], data_all["Newspaper"], data_all["Sales"], marker="o", color="blue")
axes_six.plot_surface(x_radio, y_newspaper, z_radio_newspaper_predicted.values.reshape(x_radio.shape),
                      color="red", alpha=0.2)
axes_six.set_xlabel("Radio")
axes_six.set_ylabel("Newspaper")
axes_six.set_zlabel("Sales")
axes_six.set_title("Sales - Radio + Newspaper")

print(result_radio_newspaper_sales.summary())
print("\n")

# 4. 电视, 广播, 报纸 & 销量关系
model_all = smf.ols(formula="Sales ~ TV + Radio + Newspaper", data=data_all)
result_all = model_all.fit()

x_tv, y_radio, z_newspaper = np.meshgrid(np.linspace(data_all["TV"].min(), data_all["TV"].max(), 100),
                                         np.linspace(data_all["Radio"].min(), data_all["Radio"].max(), 100),
                                         np.linspace(data_all["Newspaper"].min(), data_all["Newspaper"].max(), 100))
xyz_all = pd.DataFrame({"TV": x_tv.ravel(), "Radio": y_radio.ravel(), "Newspaper": z_newspaper.ravel()})
h_all_predicted = result_all.predict(xyz_all)

print(result_all.summary())
print("\n")

# 5. 电视, 广播 & 报纸 相关性
corr_matrix = pd.DataFrame(  # 相关系数矩阵
    {"TV": data_all["TV"].ravel(), "Radio": data_all["Radio"].ravel(), "Newspaper": data_all["Newspaper"].ravel()}
).corr()
print(corr_matrix)
print("\n")

figure_seven = plt.figure(7)  # 广播 & 电视散点图
plt.scatter(data_all["Radio"], data_all["Sales"], marker="o", color="blue", label="Radio")
plt.scatter(data_all["TV"], data_all["Sales"], marker="x", color="red", label="TV")
plt.legend()
plt.xlabel("Radio / TV")
plt.ylabel("Sales")
plt.title("Radio & TV Correlation")
figure_seven.show()

figure_eight = plt.figure(8)  # 广播 & 报纸散点图
plt.scatter(data_all["Radio"], data_all["Sales"], marker="o", color="blue", label="Radio")
plt.scatter(data_all["Newspaper"], data_all["Sales"], marker="x", color="red", label="Newspaper")
plt.legend()
plt.xlabel("Radio / Newspaper")
plt.ylabel("Sales")
plt.title("Radio & Newspaper Correlation")
figure_eight.show()

plt.show()
