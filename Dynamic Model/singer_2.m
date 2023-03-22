clear all; close all; clc;
%% initial parameter
n=6; %状态维数 ;
global T;
T=1; %采样时间
N=100; %运行总时刻
w_mu=zeros(6,1); % mean of process noise 
v_mu=[0,0,]'; % mean of measurement noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Singer模型
%covariance of process noise
global a;% 定义全局变量
a= 0.1;% Singer 机动频率
% 方差计算
global sigma_q;
sigma_q= 3;
%------------------状态噪声协方差——
q11=(1-exp(-2*a*T) + 2*a*T + 2*a^3*T^3/3 - 2*a^2*T^2 - 4*a*T*exp(-a*T) )/(2*a^5);
q12=(exp(-2*a*T)+1-2*exp(-a*T)+2*a*T*exp(-a*T)-2*a*T+a^2*T^2)/(2*a^4);
q13=(1-exp(-2*a*T)-2*a*T*exp(-a*T))/(2*a^3); 
q21=q12;
q22=(4*exp(-a*T)-3-exp(-2*a*T)+2*a*T)/(2*a^3); 
q23=(exp(-2*a*T)+1-2*exp(-a*T))/(2*a^2); 
q31=q13; 
q32=q23; 
q33=(1-exp(-2*a*T))/(2*a); 
Q=2*a*sigma_q^2*[q11, q12, q13; q21, q22, q23;  q31, q32, q33];
% % Q=blkdiag(Q,0.001);
Qk=blkdiag(Q,Q);

%%%%目标模型
F=[1 T (a*T-1+exp(-a*T))/a^2 
    0 1              (1-exp(-a*T))/a  
    0 0                      exp(-a*T)  ];
Fk=blkdiag(F,F);


% 初始状态的均值和方差
x=[0,100,10   0,200,-5 ]'; % 共8维，包括 x：位置 速度 加速度 加速度率； y:位置 速度 加速度 加速度率；
P_0=diag([1e3,10^2,10^1, 1e3,10^2,10^1]); 
x0=mvnrnd(x,P_0); % 初始状态
%x0=(x+normrnd(0,0.001)')';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 目标航迹航迹生成
x=x0';
% 航迹生成
t1=30;
t2=60;
t3=100;
for k=1:1:t1
    x(3)=10; x(6)=-5;
    w=mvnrnd(w_mu',Qk)';
    x=Fk*x + w;
    sV(:,k,1)=x;
end
for k=t1+1:1:t2
    x(3)=-10;  x(6)=20;
    w=mvnrnd(w_mu',Qk)';
    x=Fk*x + w;
    sV(:,k,1)=x;
end
for k=t2+1:1:t3
    x(3)=-1;  x(6)=-60;
    w=mvnrnd(w_mu',Qk)';
    x=Fk*x + w;
    sV(:,k,1)=x;
end
figure
plot(sV(1,:,1),sV(4,:,1),'-*r')
xlabel('m');ylabel('m');
legend('目标轨迹')
title('Singer模型目标轨迹')
xlim([-5000,21000]); % 设置坐标轴范围  
ylim([-15000,22000]);

figure
plot(sV(2,:,1),sV(5,:,1),'-*b')
xlabel('m/s');ylabel('m/s');
legend('速度轨迹')
title('Singer模型目标轨迹')
figure
plot(sV(3,:,1),sV(6,:,1),'-*g')
xlabel('m/s^2');ylabel('m/s^2');
legend('加速度轨迹')
title('Singer模型目标轨迹')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


