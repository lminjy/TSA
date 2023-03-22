clear all; close all; clc;
%% initial parameter
n=4; %dimension of the target ;

M=3; %number of rader
N=100; %the runs atime
chan=6; %channel, for the class of fiter
w_mu=[0,0]'; % mean of process noise 
v_mu=[0,0]'; % mean of measurement noise
%covariance of process noise
q_x=3; %m/s^2
q_y=q_x;
Qk=diag([q_x^2,q_y^2]); 
% state matrix
T_f=1; %sample time of fusion center
w=-pi/180*3;% 转弯角速度
Fk= [1 sin(w*T_f)/w 0 -(1-cos(w*T_f))/w
       0 cos(w*T_f)   0 -sin(w*T_f)
       0 (1-cos(w*T_f))/w 1 sin(w*T_f)/w
       0  sin(w*T_f) 0 cos(w*T_f) ]; %
Gk= [ T_f^2/2  0
       T_f     0
       0      T_f^2/2
       0      T_f ]; %
   
% intial state 
x_bar=[3500,-130,200,00]';
P_0=diag([5e6,10^4,5e6,10^4]); %initial covariance
% x_bar=[-1000,50,-800,100]';
% P_0=diag([1e5,1e3,1e5,1e3]); %initial covariance   


x0=mvnrnd(x_bar,P_0); % 初始状态
%x0=(x+normrnd(0,0.001)')';
x=x0';
for k=1:N
   %% %%%%%%% target model %%%%%%%%%%%%%%%%%%%%
   %% 目标运动学模型(被跟踪目标建模)，匀速运动CV模型
    w=mvnrnd(w_mu',Qk)';%过程噪声方差
    x=Fk*x+Gk*w;
    sV(:,k,1,1)=x;

end
% 二维匀速圆周CT运动目标轨迹       
figure
plot(sV(1,:,1,1),sV(3,:,1,1),'-*r','LineWidth',1)
grid on
xlabel('m');ylabel('m');
legend('真实轨迹')
title('匀速圆周CT运动目标轨迹')   
   
   

