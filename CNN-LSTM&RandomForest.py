import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.layers import Reshape
from tqdm import tqdm

# 读取数据
df = pd.read_csv('../Doublekalmanfused_data.csv')

# 数据预处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 定义输入和输出的时间步数
input_time_steps = 10  # 输入序列的时间步数
output_time_steps = 10  # 输出序列的时间步数
num_features = scaled_data.shape[1]  # 特征数量

X, y = [], []

for i in range(len(scaled_data) - input_time_steps - output_time_steps + 1):
    X.append(scaled_data[i:(i + input_time_steps)])
    y.append(scaled_data[(i + input_time_steps):(i + input_time_steps + output_time_steps)])

X = np.array(X)
y = np.array(y)

# 划分数据集
split_ratio = 0.8  # 80%的数据用于训练，20%用于测试

split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 构建CNN-LSTM模型
model = Sequential()

# 添加卷积层
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_time_steps, num_features)))
model.add(MaxPooling1D(pool_size=2))

# 添加LSTM层
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=32, return_sequences=True))

# 添加Flatten层
model.add(Flatten())

# 输出层
model.add(Dense(units=output_time_steps * num_features, activation='linear'))  # 输出层与输出序列的形状相匹配
model.add(Reshape((output_time_steps, num_features)))
# 编译模型
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 生成CNN-LSTM模型的训练数据的预测结果
y_train_pred = model.predict(X_train)


def rf_model(n):
    # 创建随机森林模型
    rf_model = RandomForestRegressor(n_estimators=n, random_state=64, verbose=1, n_jobs=8)

    for _ in tqdm(range(1), desc="Training Random Forest"):
        rf_model.fit(y_train_pred.reshape(-1, output_time_steps * num_features),
                     y_train.reshape(-1, output_time_steps * num_features))

    # 使用CNN-LSTM模型的测试数据进行预测
    y_test_pred_rf = rf_model.predict(model.predict(X_test).reshape(-1, output_time_steps * num_features))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, num_features))
    y_test_pred_actual_rf = scaler.inverse_transform(y_test_pred_rf.reshape(-1, num_features))

    # 计算每个样本的均方误差
    sample_mse_rf = np.mean(np.square(y_test_pred_actual_rf - y_test_actual), axis=1)

    # 计算平均均方误差
    mse_rf = np.mean(sample_mse_rf)
    print(f"Mean Squared Error (Random Forest): {mse_rf}")
    return mse_rf

#启动主函数
if __name__ == '__main__':
    n_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    mse_list = []
    for i in n_list:
        mse = rf_model(i)
        mse_list.append(mse)
    plt.figure(figsize=(12, 6))
    plt.plot(n_list, mse_list, label='MSE', linestyle='--', marker='x')
    plt.xlabel('n_estimators')
    plt.ylabel('MSE')
    plt.title('CNN-LSTM&RandomForest epoch=50')
    plt.legend()
    plt.grid(True)
    plt.show()
