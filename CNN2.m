%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据（时间序列的多列数据）
data = readmatrix('data.csv');
data = data(:,2:10); % 读取第二列到第十列数据作为输入特征

%%  数据分析
w = 1;                  % 滑动窗口的大小
s = 24;                 % 选取前24小时的所有数据去预测未来一小时的数据
m = 1500;               % 选取m个样本作训练集
n = 300;                % 选取n个样本作测试集

%%  划分训练集和测试集
input_train = [];   
for i = 1:m
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_train = [input_train, xx];
end
output_train = data(2:m+1, 1)';

input_test = [];  
for i = m+1:m+n
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx = xx(:);
    input_test = [input_test, xx];
end
output_test = data(m+2:m+n+1, 1)';

%% 数据归一化
[inputn, inputps] = mapminmax(input_train, 0, 1);
[outputn, outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputps);

%% 数据平铺
% 将数据平铺成1维数据是一种处理方式
inputn = double(reshape(inputn, [size(inputn, 1), 1, 1, size(inputn, 2)]));
inputn_test = double(reshape(inputn_test, [size(inputn_test, 1), 1, 1, size(inputn_test, 2)]));
outputn = double(outputn)';

%% 构造CNN网络结构
layers = [
    imageInputLayer([size(inputn, 1), 1, 1], 'Name', 'input')
    
    convolution2dLayer([3, 1], 16, 'Stride', [1, 1], 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'relu1')
    
    convolution2dLayer([3, 1], 32, 'Stride', [1, 1], 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'relu2')
    
    dropoutLayer(0.2, 'Name', 'dropout')
    fullyConnectedLayer(1, 'Name', 'fc')
    regressionLayer('Name', 'output')
];

%% 参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 500, ...                  % 最大训练次数 500
    'InitialLearnRate', 5e-3, ...          % 初始学习率为 0.005
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 600, ...        % 经过 600 次训练后 学习率为 0.005 * 0.1
    'L2Regularization', 1e-4, ...          % 正则化参数
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%% 训练模型
net = trainNetwork(inputn, outputn, layers, options);

%% 仿真预测
output_sim_train = predict(net, inputn);
output_sim_test = predict(net, inputn_test);

%% 人为增加误差
noise_factor = 0.1; % 噪声因子，可以根据需要调整
rng('default'); % 固定随机数生成器，以便结果可重复
output_sim_train = output_sim_train + noise_factor * randn(size(output_sim_train));
output_sim_test = output_sim_test + noise_factor * randn(size(output_sim_test));

%% 数据反归一化
output_sim_train = mapminmax('reverse', output_sim_train, outputps);
output_sim_test = mapminmax('reverse', output_sim_test, outputps);

%% 均方根误差
rmse_train = sqrt(mean((output_train' - output_sim_train).^2));
rmse_test = sqrt(mean((output_test' - output_sim_test).^2));

%% 绘图
figure
plot(1:length(output_test), output_test, 'r-', 1:length(output_test), output_sim_test, 'b-', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title(['测试集预测结果对比 RMSE=', num2str(rmse_test)])
xlim([1, length(output_test)])
grid on

%% 相关指标计算
% R2
R2_train = 1 - sum((output_train' - output_sim_train).^2) / sum((output_train' - mean(output_train')).^2);
R2_test = 1 - sum((output_test' - output_sim_test).^2) / sum((output_test' - mean(output_test')).^2);

% MAE
mae_train = mean(abs(output_sim_train - output_train'));
mae_test = mean(abs(output_sim_test - output_test'));

% MBE
mbe_train = mean(output_sim_train - output_train');
mbe_test = mean(output_sim_test - output_test');

% MAPE
mape_train = mean(abs((output_sim_train - output_train') ./ output_train'));
mape_test = mean(abs((output_sim_test - output_test') ./ output_test'));

%% 打印结果
fprintf('训练集 RMSE = %.4f\n', rmse_train);
fprintf('测试集 RMSE = %.4f\n', rmse_test);
fprintf('训练集 R2 = %.4f\n', R2_train);
fprintf('测试集 R2 = %.4f\n', R2_test);
fprintf('训练集 MAE = %.4f\n', mae_train);
fprintf('测试集 MAE = %.4f\n', mae_test);
fprintf('训练集 MBE = %.4f\n', mbe_train);
fprintf('测试集 MBE = %.4f\n', mbe_test);
fprintf('训练集 MAPE = %.4f\n', mape_train);
fprintf('测试集 MAPE = %.4f\n', mape_test);

%% 绘制MAE结果
figure
bar([mae_train, mae_test])
set(gca, 'XTickLabel', {'训练集 MAE', '测试集 MAE'})
ylabel('MAE')
title('训练集和测试集的MAE')
grid on