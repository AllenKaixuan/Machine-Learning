import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import autosklearn.classification
import sklearn.metrics
from multiprocessing import freeze_support

# 将主要代码放在函数中
def main():
    df_train = pd.read_csv('./TrainOnMe.csv')
    label_column = 'y'
    X_train = df_train.drop(columns=[label_column])
    
    y_train = df_train[label_column]
    X_train = X_train.iloc[:, 1:]
    
    # 检查标签
    print("唯一的标签值:", y_train.unique())
    
    # 标签映射
    label_mapping = {'Antrophic': 0, 'OpenAI': 1, 'Mistral': 2}
    y_train = y_train.map(label_mapping)
    
    # 处理NaN值
    valid_indices = ~y_train.isna()
    y_train = y_train[valid_indices]
    X_train = X_train.loc[valid_indices]
    
    # 处理分类特征
    X_train = pd.get_dummies(X_train)
    
    # 数据标准化
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 划分训练集和测试集
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # PCA降维
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_split)
    X_test_pca = pca.transform(X_test)
    
    # 训练神经网络
    mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, alpha=0.01, random_state=42)
    mlp.fit(X_train_pca, y_train_split)
    y_pred = mlp.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'神经网络准确率: {accuracy:.4f}')
    
    # 使用auto-sklearn (修复版本警告)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600,
        per_run_time_limit=300,
        n_jobs=-1,
        # 使用新的推荐方式设置ensemble_size
        ensemble_kwargs={'ensemble_size': 50}
    )
    
    # 训练模型
    automl.fit(X_train_pca, y_train_split)
    
    # 输出最佳模型
    print(automl.sprint_statistics())
    print("最佳模型:", automl.show_models())
    
    # 预测
    y_pred = automl.predict(X_test_pca)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"auto-sklearn准确率: {accuracy:.4f}")

# 这是解决多进程问题的关键部分
if __name__ == '__main__':
   
    main()