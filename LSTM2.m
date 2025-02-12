%% 关闭警告信息
warning off
close all
clear
clc

%% 读取数据
data = readmatrix('data.csv');
data = data(:, 2:10);  % 使用第二列到第十列数据作为输入特征

%% 数据预处理
w = 1;                 % 滑动窗口大小
s = 24;                % 前24小时的数据
m = 1500;              % 训练集样本数
n = 300;               % 测试集样本数

input_train = [];
for i = 1:m
    xx = data(1+w*(i-1):w*(i-1)+s, :);
    xx = xx(:);
    input_train = [input_train, xx];
end
output_train = data(2:m+1, 1)';

input_test = [];
for i = m+1:m+n
    xx = data(1+w*(i-1):w*(i-1)+s, :);
    xx = xx(:);
    input_test = [input_test, xx];
end
output_test = data(m+2:m+n+1, 1)';

%% 数据归一化
[inputn, inputps] = mapminmax(input_train, 0, 1);
[outputn, outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputps);

%% 转换数据格式以适应 LSTM 输入格式
inputn = reshape(inputn, [size(inputn, 1), size(inputn, 2)]);
inputn_test = reshape(inputn_test, [size(inputn_test, 1), size(inputn_test, 2)]);

%% 转换数据为元胞数组格式
inputn_cell = mat2cell(inputn, size(inputn, 1), ones(1, size(inputn, 2)));
inputn_test_cell = mat2cell(inputn_test, size(inputn_test, 1), ones(1, size(inputn_test, 2)));

%% LSTM模型构建
numFeatures = size(inputn, 1);
numResponses = 1;
numHiddenUnits = 10;

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 5e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 250, ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 1e-4, ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 训练LSTM模型
net = trainNetwork(inputn_cell, outputn', layers, options);

%% 进行预测
t_sim1 = predict(net, inputn_cell);
t_sim2 = predict(net, inputn_test_cell);

%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, outputps);
T_sim2 = mapminmax('reverse', t_sim2, outputps);

%% 计算误差
error1 = sqrt(sum((T_sim1' - output_train).^2) ./ m);
error2 = sqrt(sum((T_sim2' - output_test).^2) ./ n);

%% 显示网络结构
analyzeNetwork(net)

%% 绘图
figure
plot(1:n, output_test, 'r-', 1:n, T_sim2', 'b-', 'LineWidth', 1)
legend('实际值', '预测值')
xlabel('预测样本数')
ylabel('预测值')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, n])
grid

%% 计算评价指标
% R2
R1 = 1 - norm(output_train - T_sim1')^2 / norm(output_train - mean(output_train))^2;
R2 = 1 - norm(output_test - T_sim2')^2 / norm(output_test - mean(output_test))^2;

disp(['测试集数据的R2为：', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1' - output_train)) ./ m;
mae2 = sum(abs(T_sim2' - output_test)) ./ n;

disp(['测试集数据的MAE为：', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1' - output_train) ./ m;
mbe2 = sum(T_sim2' - output_test) ./ n;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

% MAPE
mape1 = sum(abs((T_sim1' - output_train)./output_train)) ./ m;
mape2 = sum(abs((T_sim2' - output_test)./output_test)) ./ n;

disp(['训练集数据的MAPE为：', num2str(mape1)])
disp(['测试集数据的MAPE为：', num2str(mape2)])
