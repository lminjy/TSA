%% 二维匀速CV运动目标轨迹       
clc;
clear all;
close all;
n=4; % state dimension : 6
T=1; % sample time.
N=100; %the runs atime，跟踪总时长
w_mu=[0,0]';
%% target model
q=3; % 目标运动学标准差，过程噪声
Qk=q^2*eye(2);% cov. of process noise
% state matrix
Fk= [1 T 0 0
     0 1 0 0
     0 0 1 T
     0 0 0 1 ]; %
Gk= [ T^2/2 0 
       T 0     
      0 T^2/2  
      0 T  ]; %
   
   
% 
%% define parameter
sV=zeros(n,N,1,1); % state
x=[1000,20,1000,20]';
P_0=diag([1e5,10^2,1e5, 10^2]); 

x0=mvnrnd(x,P_0); % 初始状态
%x0=(x+normrnd(0,0.001)')';
x=x0';
for k=1:N
   %% %%%%%%% target model %%%%%%%%%%%%%%%%%%%%
   %% 目标运动学模型(被跟踪目标建模)，匀速运动CV模型
    w=mvnrnd(w_mu',Qk)';%过程噪声方差
    x=Fk*x+Gk*w;
    sV(:,k,1,1)=x;

end
% 二维匀速CV运动目标轨迹 
ii=1:N;
figure
plot(sV(1,:,1,1),sV(3,:,1,1),'-*r','LineWidth',1)
grid on
xlabel('m');ylabel('m');
legend('真实轨迹')
title('二维维匀速运动目标轨迹')




