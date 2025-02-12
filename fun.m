function [net,mse0] = fun(x,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize)
%函数用于计算粒子适应度值

setdemorandstream(pi);
%rng default;%固定随机数
numFilters = fix(x(3));
filterSize = fix(x(4));
dropoutFactor = 0.005;
numBlocks =  fix(x(5));

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
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 500, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', x(1), ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod',300, ...                   % 训练100次后开始调整学习率
    'LearnRateDropFactor',0.001, ...                    % 学习率调整因子
    'L2Regularization', x(2), ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

% 网络训练
net = trainNetwork(inputn,outputn,lgraph,options );
an = net.predict(inputn_test);
test_simu  = mapminmax('reverse',an,outputps);
mse0=mse(output_test,test_simu);
end