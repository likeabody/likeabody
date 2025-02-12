%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  �������ݣ�ʱ�����еĵ������ݣ�
result = xlsread('���ݼ�.xlsx');

%%  ���ݷ���
num_samples = length(result);  % �������� 
kim = 15;                      % ��ʱ������kim����ʷ������Ϊ�Ա�����
zim =  1;                      % ��zim��ʱ������Ԥ��

%%  �������ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end
%%  ���ݼ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��
% %%  ����ѵ�����Ͳ��Լ�
% temp = 1: 1: 922;
% 
% P_train = res(temp(1: 700), 1: 15)';
% T_train = res(temp(1: 700), 16)';
% M = size(P_train, 2);
% 
% P_test = res(temp(701: end), 1: 15)';
% T_test = res(temp(701: end), 16)';
% N = size(P_test, 2);
%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
% ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
% Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
% ����Ӧ��ʼ�պ���������ݽṹ����һ��
P_train =  double(reshape(P_train, f_, 1, 1, M));
P_test  =  double(reshape(P_test , f_, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  ���ݸ�ʽת��
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  ����ģ��
layers = [
    sequenceInputLayer(f_)              % ���������
    
    lstmLayer(10, 'OutputMode', 'last') % LSTM��
    reluLayer                           % Relu�����
    
    fullyConnectedLayer(1)              % ȫ���Ӳ�
    regressionLayer];                   % �ع��

%%  ��������
% options = trainingOptions('adam', ...       % Adam �ݶ��½��㷨
%     'MaxEpochs', 1200, ...                  % ���ѵ������
%     'InitialLearnRate', 5e-3, ...           % ��ʼѧϰ��
%     'LearnRateSchedule', 'piecewise', ...   % ѧϰ���½�
%     'LearnRateDropFactor', 0.1, ...         % ѧϰ���½�����
%     'LearnRateDropPeriod', 800, ...         % ���� 800 ��ѵ���� ѧϰ��Ϊ 0.005 * 0.1
%     'Shuffle', 'every-epoch', ...           % ÿ��ѵ���������ݼ�
%     'Plots', 'training-progress', ...       % ��������
%     'Verbose', false);
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 300, ...                             % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', 5e-3, ...                     % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 250, ...                   % ѵ��250�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.1, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 1e-4, ...                     % ���򻯲���
    'ExecutionEnvironment', 'auto',...                % ѵ������
    'Verbose', false, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������
%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  ����Ԥ��
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  �鿴����ṹ
analyzeNetwork(net)

%%  ��ͼ
figure
plot(1: M, T_train, 'r-', 1: M, T_sim1, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-', 1: N, T_sim2, 'b-', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])

%  MAPE
mape1 = sum(abs((T_sim1' - T_train)./T_train)) ./ M ;
mape2 = sum(abs((T_sim2' - T_test )./T_test )) ./ N ;

disp(['ѵ�������ݵ�MAPEΪ��', num2str(mape1)])
disp(['���Լ����ݵ�MAPEΪ��', num2str(mape2)])

%%  ����ɢ��ͼ
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('ѵ������ʵֵ');
ylabel('ѵ����Ԥ��ֵ');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('ѵ����Ԥ��ֵ vs. ѵ������ʵֵ')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('���Լ���ʵֵ');
ylabel('���Լ�Ԥ��ֵ');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('���Լ�Ԥ��ֵ vs. ���Լ���ʵֵ')
%%%%%%%%%����ţ�ĬĬ������