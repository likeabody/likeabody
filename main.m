clear
close all
clc
data =  readmatrix('data.csv');
data = data(:,2:10);
w=1;                  % w是滑动窗口的大小
s=24;                  % 选取前24小时的所有数据去预测未来一小时的数据
m = 1500;            %选取m个样本作训练集
n = 500;             %选取n个样本作测试集
input_train=[];   
for i =1:m
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx =xx(:);
    input_train = [input_train,xx];
end
output_train =[];  
output_train = data(2:m+1,1)';

input_test=[];  
for i =m+1:m+n
    xx = data(1+w*(i-1):w*(i-1)+s,:);
    xx =xx(:);
    input_test = [input_test,xx];
end
output_test = data(m+2:m+n+1,1)';

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);


%% 优化算法优化前，构建优化前的TCN模型
numFeatures = size(input_test,1);
outputSize = 1;  %数据输出y的维度  
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 4;

layer = sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor) 
        % spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor) 
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(outputSize,Name="fc")
    tanhLayer('name','softmax')
    regressionLayer('name','output')];
    
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc");

%  参数设置
options0 = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 500, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.001, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod',300, ...                   % 训练100次后开始调整学习率
    'LearnRateDropFactor',0.001, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

% 网络训练
net0 = trainNetwork(inputn,outputn,lgraph,options0 );
an0 = net0.predict(inputn_test);
test_simu0=mapminmax('reverse',an0,outputps); %把仿真得到的数据还原为原始的数量级
%误差指标
error0 = output_test - test_simu0;
mse0=mse(output_test,test_simu0)
%% 标准TCN神经网络作图
figure
plot(output_test,'b-','markerfacecolor',[0.5,0.5,0.9],'MarkerSize',6)
hold on
plot(test_simu0,'r--','MarkerSize',6)
title(['TCN的mse误差：',num2str(mse0)])
legend('真实y','预测的y')
xlabel('样本数')
ylabel('负荷值')
box off
set(gcf,'color','w')

%% 调用DBO优化TCN
disp('调用DBO优化TCN......,优化时间较长，请耐心等待，因为需要训练SearchAgents*Max_iterations次的TCN网络')
% DBO优化参数设置
SearchAgents = 30; % 种群数量  30
Max_iterations = 20; % 迭代次数   20
lowerbound = [0.001 0.0001 20 2 2]; %五个参数的下限分别是正则化参数，学习率，滤波器个数:numFilters，滤波器大小:filterSize，区块数:numBlocks
upperbound = [0.1 0.01 100 10 10];    %五个参数的上限
dimension = length(lowerbound);%数量，即要优化的TCN参数个数
[fMin,Best_pos,Convergence_curve,bestnet]  = DBOforTCN(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);

L2Regularization = Best_pos(1,1); % 最佳L2正则化系数
InitialLearnRate = Best_pos(1,2) ;% 最佳初始学习率
numFilters = fix(Best_pos(1,3)); 
filterSize = fix(Best_pos(1,4));
numBlocks = fix(Best_pos(1,5));
disp(['最优参数：',num2str(L2Regularization),num2str(InitialLearnRate),num2str(numFilters),num2str(filterSize),num2str(numBlocks)])
disp('优化结束，将最佳net导出并用于测试......')
setdemorandstream(pi);
%% 对测试集的测试
an = bestnet.predict(inputn_test); 
test_simu  = mapminmax('reverse',an,outputps);
error = test_simu-output_test;
msee = mse(output_test,test_simu);
figure
hold on 
plot(test_simu,'g')
plot(output_test,'b')
legend('DBO-TCN预测值','实际值')
ylabel('预测结果')
xlabel('预测样本')
title(['DBO-TCN的mse误差：',num2str(msee)])
box off
set(gcf,'color','w')

figure
plot(abs(error),'g-*')
hold on
plot(abs(error0),'-or')
title('预测误差对比图','fontsize',12)
xlabel('预测样本','fontsize',12)
ylabel('误差绝对值','fontsize',12)
legend('DBO-TCN','TCN')


%% 回归图与误差直方图
figure;
plotregression(test_simu,output_test,['优化后回归图']);
set(gcf,'color','w')

figure;
ploterrhist(test_simu-output_test,['误差直方图']);
set(gcf,'color','w')

%% 打印出评价指标
% 预测结果评价
ae= abs(test_simu-output_test);
rmse = (mean(ae.^2)).^0.5;
mse = mean(ae.^2);
mae = mean(ae);
mape = mean(ae./test_simu);
[R,r] = corr(output_test,test_simu);
R2 = 1 - norm(output_test -  test_simu)^2 / norm(output_test-mean(output_test ))^2;
disp('预测结果评价指标：')
disp(['RMSE = ', num2str(rmse)])
disp(['MSE  = ', num2str(mse)])
disp(['MAE  = ', num2str(mae)])
disp(['MAPE = ', num2str(mape)])
disp(['决定系数R^2为：  ',num2str(R2)])