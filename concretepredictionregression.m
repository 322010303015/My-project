close all;
clc;
A=readmatrix('concrete.csv');
x=A(:,1:8)';
t=A(:,9)';
trainFcn='traingd';
hiddenlayersize=10;
net = feedforwardnet(hiddenlayersize,trainFcn);
net.divideParam.trainRatio=70/100;
net.divideParam.valRatio=15/100;
net.divideParam.testRatio=15/100;
[net,tr]=train(net,x,t);
y=net(x);
e=gsubtract(t,y);
performance=perform(net,t,y);
view(net)