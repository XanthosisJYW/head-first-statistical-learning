import pandas as pd
import pydotplus
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 读取数据
balance_data = pd.read_csv("../data/BalanceScale.csv")

# 2. 分别提取属性 & 标签
attributes = balance_data.values[:, 1:5]
label = balance_data.values[:, 0]

# 3. 切分训练集 & 验证集
x_train, x_validate, y_train, y_validate = train_test_split(attributes, label, test_size=0.2, random_state=100)

# 4. 用训练集生成决策树 & 输出到文件
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                     max_depth=3, min_samples_leaf=5)
model = clf_entropy.fit(x_train, y_train)

with open("decision_tree.txt", "w") as file:
    file = tree.export_graphviz(model, out_file=file)
    # http://webgraphviz.com/

print(model)
print("\n")

# 5. 用检验集测试准确率
y_predict = clf_entropy.predict(x_validate)
# print(y_predict)
# print("\n")
print("Model Accuracy: ", accuracy_score(y_validate, y_predict))
