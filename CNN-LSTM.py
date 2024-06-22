import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.layers import Reshape
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 读取数据
df = pd.read_csv('../kalmanfused_data.csv')

# 数据预处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 定义输入和输出的时间步数
input_time_steps = 10  # 输入序列的时间步数
output_time_steps = 10  # 输出序列的时间步数
num_features = scaled_data.shape[1]  # 特征数量
print(num_features)
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


def model(epochs):
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
    # 调整输出层
    model.add(Dense(units=output_time_steps * num_features, activation='linear'))
    model.add(Reshape((output_time_steps, num_features)))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    # 模型评估
    y_pred = model.predict(X_test)

    # 反向缩放整个数据集
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, num_features))
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, num_features))

    # 计算每个样本的均方误差
    sample_mse = np.mean(np.square(y_pred_actual - y_test_actual), axis=1)

    # 计算平均均方误差
    mse = np.mean(sample_mse)

    print(f"Mean Squared Error: {mse}")
    return mse


if __name__ == '__main__':
    epoch_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    mse_list = []
    for i in epoch_list:
        mse = model(i)
        mse_list.append(mse)
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_list, mse_list, label='MSE', linestyle='--', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('CNN-LSTM')
    plt.legend()
    plt.grid(True)
    plt.show()
