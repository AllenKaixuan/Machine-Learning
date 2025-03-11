import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


df_train = pd.read_csv('./TrainOnMe.csv')
label_column = 'y'
X_train = df_train.drop(columns=[label_column])

y_train = df_train[label_column]
X_train = X_train.iloc[:, 1:]
X_train = pd.get_dummies(X_train)




label_mapping = {'Antrophic': 0, 'OpenAI': 1, 'Mistral': 2}
y_train = y_train.map(label_mapping)


scaler =RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=42)



mlp = MLPClassifier(hidden_layer_sizes=(300, 50,20), max_iter=200, alpha=2.2, solver='adam', random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')





df_eval = pd.read_csv('./EvaluateOnMe.csv')

# Handle categorical features in the evaluation dataset by using the same columns as in the training dataset
X_eval = pd.get_dummies(df_eval)


X_eval = X_eval.iloc[:, 1:]
print(X_eval.columns)



X_eval_scaled = scaler.transform(X_eval)


eval_predictions = mlp.predict(X_eval_scaled)


reverse_label_mapping = {v: k for k, v in label_mapping.items()}
eval_predictions_labels = [reverse_label_mapping[pred] for pred in eval_predictions]


with open('./predicted_labels.txt', 'w') as f:
    for label in eval_predictions_labels:
        f.write(f"{label}\n")

print("Predictions saved to 'predicted_labels.txt'.")