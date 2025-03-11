import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import autosklearn.classification
import sklearn.metrics


df_train = pd.read_csv('./TrainOnMe.csv')
label_column = 'y'
X_train = df_train.drop(columns=[label_column])

y_train = df_train[label_column]
X_train = X_train.iloc[:, 1:]


# print("Unique labels:", y_train.unique())


label_mapping = {'Antrophic': 0, 'OpenAI': 1, 'Mistral': 2}
y_train = y_train.map(label_mapping)



X_train = pd.get_dummies(X_train)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=42)

pca = PCA(n_components=0.85)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(300, 50, 20), max_iter=300, alpha=1.2, solver='adam', random_state=42)
mlp.fit(X_train_pca, y_train)
y_pred = mlp.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')



# 创建自动机器学习分类器
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 搜索时间限制(秒)
    per_run_time_limit=300,  # 每次运行时间限制(秒)
    n_jobs=-1,
    ensemble_size=50
)

# 训练模型
automl.fit(X_train_pca, y_train)

# 输出最佳模型
print(automl.sprint_statistics())
print("最佳模型:", automl.show_models())

# 预测
y_pred = automl.predict(X_test_pca)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")