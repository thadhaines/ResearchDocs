%%  m410_comparison.m
%   Thad Haines         m410
%   Program Purpose:    comparison of methods calculations and plot

%
%   History:
%   04/17/19    13:24   init from m410_eulerHW
%   04/17/19    13:26   added comparison graph
%   04/22/19    11:51   init from ralstonHW
%   04/18/20    19:40   Adaption for thesis appendix scratch

%% init
clear; format compact; clc; close all;

%% Knowns
ic = [0,0];    % initial x,y contditions
intervalLength = 1;
n = 8; % <- Must always be EVEN for hw output to work

%% Function specific inlines
%trig
f = @(x,y) -sin(2*pi*x)
fp = @(x,y) -2*pi*cos(2*pi*x)
findC = @(x,y) y+2*pi*cos(2*pi*x) 

% % %exp
A = 1
intervalLength = 2;
f = @(x,c) A*exp(A*x)+c
fp = @(x,y) exp(A*x)
findC = @(x,y) y-A*exp(A*x) 

% % %log
A = 10
ic=[1,1]
f = @(x,c) A*log(x)+c
fp = @(x,y) A./x
findC = @(x,y) y-A*log(x)

%% approximations
interval = [ic(1),ic(1)+intervalLength];
t= linspace(interval(1),interval(2),n*100); % exact solution has 10n points
c = findC(ic(1), ic(2));

[yAB,xAB] = ab(fp,interval(1),ic(2),interval(2),n);
[yE,xE] = Euler(fp,interval(1),ic(2),interval(2),n);
[yrk45, xrk45] = rk4(fp,interval(1),ic(2),interval(2),n);

exact = f(interval(2),c);

%% integration using trapezoidal method

intE = trapezoidal(xE,yE)
intRK = trapezoidal(xrk45, yrk45)
intAB = trapezoidal(xAB,yAB)
intExact = trapezoidal(t, f(t,c))

%% Plotting
figure
plot(t,f(t,c),'k.-')
hold on
plot(xE,yE,'bo:')
plot(xrk45,yrk45,'mp:','markersize',10)
plot(xAB,yAB,'gs:','markersize',10)

grid on
legend('Exact f(x)','Euler','Runge Kutta 45','Adams-Bashforth','location','best')
set(gca, 'fontsize',13)
xlim(interval)
%set(gcf,'position',[1936 448 560 420])