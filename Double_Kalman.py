import numpy as np
import pandas as pd


# 初始化卡尔曼滤波器
def initialize_KalmanFilter(initial_state_estimate, initial_state_covariance):
    state_estimate = initial_state_estimate
    state_covariance = initial_state_covariance
    return state_estimate, state_covariance


# 预测步骤
def predict(state_estimate, state_covariance, A, Q):
    predicted_state_estimate = np.dot(A, state_estimate)
    predicted_state_covariance = np.dot(np.dot(A, state_covariance), A.T) + Q
    return predicted_state_estimate, predicted_state_covariance


# 更新步骤
def update(predicted_state_estimate, predicted_state_covariance, H, R, measurement):
    kalman_gain = np.dot(np.dot(predicted_state_covariance, H.T),
                         np.linalg.inv(np.dot(np.dot(H, predicted_state_covariance), H.T) + R))
    updated_state_estimate = predicted_state_estimate + np.dot(kalman_gain,
                                                               (measurement - np.dot(H, predicted_state_estimate)))
    updated_state_covariance = np.dot((np.eye(len(predicted_state_estimate)) - np.dot(kalman_gain, H)),
                                      predicted_state_covariance)
    return updated_state_estimate, updated_state_covariance


# 融合数据
def fuse_data(state_estimate1, state_covariance1, state_estimate2, state_covariance2, weight1, weight2):
    fused_state_estimate = weight1 * state_estimate1 + weight2 * state_estimate2
    fused_state_covariance = weight1 * state_covariance1 + weight2 * state_covariance2
    return fused_state_estimate, fused_state_covariance


# 主循环
def main():
    data_group1 = pd.read_csv('../data_1.csv')
    data_group2 = pd.read_csv('../data_2.csv')

    first_row_data1 = data_group1.iloc[0].values
    second_row_data1 = data_group1.iloc[1].values

    # 计算第一行和第二行数据之间的差异
    covariance_diag = np.abs(second_row_data1 - first_row_data1)

    # 创建初始状态协方差矩阵
    initial_state_covariance1 = np.diag(covariance_diag)

    # 设置初始状态估计
    initial_state_estimate1 = first_row_data1

    first_row_data2 = data_group2.iloc[0].values
    second_row_data2 = data_group2.iloc[1].values

    # 计算第一行和第二行数据之间的差异
    covariance_diag = np.abs(second_row_data2 - first_row_data2)

    # 创建初始状态协方差矩阵
    initial_state_covariance2 = np.diag(covariance_diag)

    # 设置初始状态估计
    initial_state_estimate2 = first_row_data2
    A = np.eye(27)  # 状态转移矩阵
    Q = 0.5 * np.eye(27)  # 过程噪声协方差矩阵
    H = 0.7 * np.eye(27)  # 观测矩阵
    R = 2 * np.eye(27)  # 观测噪声协方差矩阵
    weight1 = 0.5
    weight2 = 0.5

    state_estimate1, state_covariance1 = initialize_KalmanFilter(initial_state_estimate1, initial_state_covariance1)
    state_estimate2, state_covariance2 = initialize_KalmanFilter(initial_state_estimate2, initial_state_covariance2)
    updated_covariance1 = initial_state_covariance1  # 初始化更新后的协方差矩阵1
    updated_covariance2 = initial_state_covariance2  # 初始化更新后的协方差矩阵2

    fused_data_list = []  # 用于保存融合后的数据

    for index, row in data_group1.iterrows():
        measurement_group1 = row.values  # 获取传感器数据
        measurement_group2 = data_group2.iloc[index].values  # 与data_group1对应的传感器数据

        # 预测和更新步骤 - 传感器组1
        predicted_state1, updated_covariance1 = predict(state_estimate1, updated_covariance1, A, Q)
        updated_state1, updated_covariance1 = update(predicted_state1, updated_covariance1, H, R, measurement_group1)

        # 预测和更新步骤 - 传感器组2
        predicted_state2, updated_covariance2 = predict(state_estimate2, updated_covariance2, A, Q)
        updated_state2, updated_covariance2 = update(predicted_state2, updated_covariance2, H, R, measurement_group2)

        # 更新状态估计
        state_estimate1 = updated_state1
        state_estimate2 = updated_state2

        # 融合数据
        fused_state, _ = fuse_data(updated_state1, updated_covariance1, updated_state2, updated_covariance2, weight1,
                                   weight2)

        # 将融合后的数据添加到列表中
        fused_data_list.append(fused_state)

        # 输出融合后的数据
        print(f"Time Step {index + 1}: Fused Data - {fused_state}")

    # 创建一个DataFrame来保存融合后的数据
    fused_data = pd.DataFrame(fused_data_list)

    # 将DataFrame保存为CSV文件
    fused_data.to_csv('Doublekalmanfused_data.csv', index=False)  # 保存为CSV文件，不包含行索引


if __name__ == '__main__':
    main()
