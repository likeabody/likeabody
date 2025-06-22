%% I. 清空环境变量
clear all
clc
 
%% II. 训练集/测试集产生
%%
% 1. 导入数据
load pH_42.mat

%%
% 2. 产生训练集和测试集

% 训练集――900个样本
P_train = pH_42((1:900),1:4)';
T_train = pH_42((1:900),5:6)';

% 测试集――100个样本
P_test = pH_42((901:999),1:4)';
T_test = pH_42((901:999),5:6)';
N = size(P_test,2);

%% III. 数据归一化
%%
% 1. 训练集
[Pn_train,inputps] = mapminmax(P_train);
Pn_test = mapminmax('apply',P_test,inputps);

%%
% 2. 测试集
[Tn_train,outputps] = mapminmax(T_train);
Tn_test = mapminmax('apply',T_test,outputps);

%% IV. ELM创建/训练
tic;
[IW,B,LW,TF,TYPE] = elmtrain(Pn_train,Tn_train,30,'sig',0);

%% V. ELM仿真测试
tn_sim = elmpredict(Pn_test,IW,B,LW,TF,TYPE);
toc;
%%
% 1. 反归一化
T_sim = mapminmax('reverse',tn_sim,outputps);
T_sim1 = T_sim(1,:);
T_sim2 = T_sim(2,:);

%% VI. 结果对比
T_test1 = T_test(1,:);
T_test2 = T_test(2,:);
result = [T_test' T_sim'];

%%
% 1. 均方误差
% E = mse(T_sim - T_test);
E1 = mse(T_sim1 - T_test1);
E2 = mse(T_sim2 - T_test2);
%%
% 2. 决定系数
N = length(T_test);
% R2=(N*sum(T_sim.*T_test)-sum(T_sim).*sum(T_test)).^2/((N*sum((T_sim).^2)-(sum(T_sim)).^2).*(N*sum((T_test).^2)-(sum(T_test)).^2)); 
R12 = (N*sum(T_sim1.*T_test1)-sum(T_sim1).*sum(T_test1)).^2/((N*sum((T_sim1).^2)-(sum(T_sim1)).^2).*(N*sum((T_test1).^2)-(sum(T_test1)).^2)); 
R22 = (N*sum(T_sim2.*T_test2)-sum(T_sim2).*sum(T_test2)).^2/((N*sum((T_sim2).^2)-(sum(T_sim2)).^2).*(N*sum((T_test2).^2)-(sum(T_test2)).^2)); 

 
% % VII. 绘图
% %单pH输出
% figure(1)
% plot(1:N,T_test,'r-*',1:N,T_sim,'b:o')
% grid on
% legend('真实值','预测值')
% xlabel('样本编号')
% ylabel('pH值')
% % string = {'预测结果对比(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
% title(string)

% figure(1)
% plot(1:N,T_test1,'r-*',1:N,T_sim1,'b:o')
% grid on
% legend('真实值','预测值')
% xlabel('样本编号')
% ylabel('液位值')
% string1 = {'液位值预测结果对比(ELM)';['(mse = ' num2str(E1) ' R^2 = ' num2str(R12) ')']};
% title(string1)

figure(2)
plot(1:N,T_test2,'r-*',1:N,T_sim2,'b:o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('pH值')
string2 = {'pH预测结果对比(ELM)';['(mse = ' num2str(E2) ' R^2 = ' num2str(R22) ')']};
title(string2)


