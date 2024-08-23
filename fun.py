import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载CSV文件
df = pd.read_csv('data_music/features_30_sec.csv')

# 描述性统计分析
print(df.describe())

# 可视化分析
# 假设'feature1'是你要分析的一个特征
sns.boxplot(x='label', y='feature1', data=df)
plt.show()

# 计算相关系数（如果label是数值型）
correlation = df.corr()
print(correlation['label'])  # 假设'label'是最后一列的列名

# 特征重要性评估（以随机森林为例）
from sklearn.ensemble import RandomForestClassifier

# 假设最后一列的列名是'label'
X = df.iloc[:, 1:-1]  # 特征列
y = df.iloc[:, -1]   # 分类标签

model = RandomForestClassifier()
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))

# 绘制特征重要性
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.show()