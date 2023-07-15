clear
close all

% Set problem, grid & options
clear prb grd
grd.Nx{1} = 101;
grd.Xn{1}.hi = 2*pi;
grd.Xn{1}.lo = -2*pi;
grd.Nx{2}    = 101;
grd.Xn{2}.hi = 8;
grd.Xn{2}.lo = -8;
grd.Nu{1}    = 41;
grd.Un{1}.hi = 2;
grd.Un{1}.lo = -2;
grd.X0{1} = -3+2*pi;
% grd.X0{1} = 3;
grd.XN{1}.hi = 0.02;
grd.XN{1}.lo = -0.02;
grd.X0{2} = 0;
grd.XN{2}.hi = 8;
grd.XN{2}.lo = -8;

prb.Ts = 0.05;
prb.N  = 200;
prb.W{1} = (1:prb.N)';

options = dpm();
options.MyInf = 1e+5;
options.BoundaryMethod = 'none'; % also possible: 'none' or 'LevelSet';
if strcmp(options.BoundaryMethod,'Line')
    %these options are only needed if 'Line' is used
    options.Iter = 10; %5
    options.Tol = 1e-8; %8
    options.FixedGrid = 1; %0
end

tStart = tic;
[res dyn] = dpm(@pendulum,[],grd,prb,options);
tEnd = toc(tStart)

%%
figure(1)
plot(res.x1)
hold on
plot(res.x2)
hold off

figure(2)
plot(res.u)

Cul_reward = -sum(res.reward)
Cul_rewardDis = -sum(res.rewardDis)

figure(3)
plot(res.reward)
