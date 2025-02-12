function [net,mse0] = fun(x,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize)
%�������ڼ���������Ӧ��ֵ

setdemorandstream(pi);
%rng default;%�̶������
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

%  ��������
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 500, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', x(1), ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod',300, ...                   % ѵ��100�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.001, ...                    % ѧϰ�ʵ�������
    'L2Regularization', x(2), ...         % ���򻯲���
    'ExecutionEnvironment', 'gpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'none');                    % ��������

% ����ѵ��
net = trainNetwork(inputn,outputn,lgraph,options );
an = net.predict(inputn_test);
test_simu  = mapminmax('reverse',an,outputps);
mse0=mse(output_test,test_simu);
end