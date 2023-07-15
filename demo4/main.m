clear
close all

%% USER CONSTANT
max_step = 200;

x0 = [3;0];
u0 = 0;

SAVE_PLOT = 1;

Np = 50; Nc =Np;

TRAIN_NAME = "test_train14";

NN_NAME = [
    "DP1"
    "DP2"
%     "LQR"
%     "MPC"    
    "NLMPC"
     "eps0.99_gamma1_0_0_end"
     "eps0.75_gamma1_0_0_end"
     "eps0.50_gamma1_0_0_end"
     "eps0.25_gamma1_0_0_end"
     "eps0.00_gamma1_0_0_end"
     "eps0.99_gamma0.75_0_0_end"
     "eps0.99_gamma0.50_0_0_end"
     "eps0.99_gamma0.25_0_0_end"
     "eps0.99_gamma0.00_0_0_end"
     "eps0.99_gamma0.975_0_0_end"
     "eps0.99_gamma0.950_0_0_end"
     "eps0.99_gamma0.925_0_0_end"
     "eps0.99_gamma0.900_0_0_end"
     ];

% data_legent = NN_NAME;
data_legend = [
    "DP to 2\pi"
    "DP to 0"
    "NLMPC"
     "DQN (\epsilon: 0.99)"
     "DQN (\epsilon: 0.75)"
     "DQN (\epsilon: 0.50)"
     "DQN (\epsilon: 0.25)"
     "DQN (\epsilon: 0.00)"
     "DQN (\gamma: 0.75)"
     "DQN (\gamma: 0.50)"
     "DQN (\gamma: 0.25)"
     "DQN (\gamma: 0.00)"
     "DQN (\gamma: 0.975)"
     "DQN (\gamma: 0.950)"
     "DQN (\gamma: 0.925)"
     "DQN (\gamma: 0.900)"     
    ];

%% SELECTED CASES
plot_names = ["CONVENTIONAL", "VARIOUS_EXPLORATION", "VARIOUS_GAMMA", "GAMMA_from_0.9_to_1"];
selected0 = [1 2 3];
selected1 = [1 4 5 6 7 8];
selected2 = [1 4 9 10 11 12];
selected3 = [1 4 13 14 15 16 ];

%% CONSTANTS
case_num = size(NN_NAME, 1);

traj_list = [repmat(x0, [case_num, 1]) zeros(2*case_num, max_step-1)];
u_list = zeros(case_num, max_step);
t_list = zeros(case_num, max_step);
r_sum = zeros(case_num, 1);

dt = 0.05;

%% CONVENTIOANL CONTROL
[A,B,Q,R] = pen();
sysd = ss(A,B,eye(2),zeros(2,1),dt);
sys = d2c(sysd);

% LQR ================================
tic
[K,S,P] = lqr(sys,Q,R);
lqr_t = toc;
% MPC ================================
% [A,B,nominal] = adapPen(x0, u0);
% sysd = ss(A,B,eye(2),zeros(2,1),dt);
% sys = d2c(sysd);
% 
% mpcobj = mpc(sysd, Np, Nc);
% 
% mpcobj = mpc(sysd, dt, Np, Nc);
% ms = mpcstate(mpcobj);
% 
% mpcobj.OutputVariables(2).Min = -8;
% mpcobj.OutputVariables(2).Max = +8;
% mpcobj.ManipulatedVariables.Min = -2;
% mpcobj.ManipulatedVariables.Max = +2;
% mpcobj.Weights.ManipulatedVariables = 1e-3;
% mpcobj.Weights.OutputVariables = [1 0.1];
% mpcobj.Weights.ManipulatedVariablesRate = 0;


% Y = struct('Weight',[5,0.1],'Min',[-0,-0],'Max',[0,0], 'MinECR', [0,0], 'MaxECR', [0,0]);
% U = struct('Min',-2,'Max',2);
% setterminal(mpcobj,Y,U)


% mpcobj.OutputVariables(1).MinECR(end) = 0;
% mpcobj.OutputVariables(1).MaxECR(end) = 0;
% mpcobj.OutputVariables(2).MinECR(end) = 0;
% mpcobj.OutputVariables(2).MaxECR(end) = 0;

% NLMPC ==============================

nlobj = nlmpc(2, 2, 1);

nlobj.Ts = dt;
nlobj.PredictionHorizon = Np;
nlobj.ControlHorizon = Nc;
nlobj.OutputVariables(2).Min = -8;
nlobj.OutputVariables(2).Max = +8;
nlobj.ManipulatedVariables.Min = -2;
nlobj.ManipulatedVariables.Max = +2;
nlobj.Weights.ManipulatedVariables = 1e-3;
nlobj.Weights.OutputVariables = [1 0.1];
nlobj.Weights.ManipulatedVariablesRate = 0;

nlobj.Model.StateFcn = "stateFun";
nlobj.Model.OutputFcn = @(x,u) x;
validateFcns(nlobj, x0, u0);

% DP ================================
load res
u1 = res.u;
load res3
u2 = res.u;
%% DEMOSTRAION

for j = 1:1:case_num
    x = x0; u = u0;
    if NN_NAME(j) ~= "DP1" && NN_NAME(j) ~= "DP2" && NN_NAME(j) ~= "LQR" && NN_NAME(j) ~= "MPC" && NN_NAME(j) ~= "NLMPC"
        NN_PATH = "./onnx/" + TRAIN_NAME + "/" + NN_NAME(j) + ".onnx";
        nn = importONNXNetwork(NN_PATH, TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC");
    end
    r = 0;

    for k = 1:1:max_step
        tic
        if NN_NAME(j) == "DP1"
            u = -u1(k);
        elseif NN_NAME(j) == "DP2"
            u = u2(k);
        elseif NN_NAME(j) == "LQR"
            u = -K*x;
        elseif NN_NAME(j) == "MPC"
            [A,B,nominal] = adapPen(x, u);
            sysd = ss(A,B,eye(2),zeros(2,1),dt);
            sys = d2c(sysd);

            [u, info] = mpcmoveAdaptive(mpcobj, mpcstate(mpcobj), sysd, nominal, x, [0;0]);
%             u = mpcmove(mpcobj, mpcstate(mpcobj), x, [0;0]);
        elseif NN_NAME(j) =="NLMPC"
            u = nlmpcmove(nlobj, x, u);
        else
            u = select_action(x, nn);
        end
        t = toc;
        
        x = step(x, u);
       
	x_norm = [rem(x(1)+pi, 2*pi) - pi; x(2)];

	r = x_norm'*Q*x_norm + u'*R*u;

        traj_list((1:2) + 2*(j-1), k) = x;
        u_list(j, k) = u;
        t_list(j, k) = t;
	r_sum(j, 1) = r_sum(j, 1) + r ;
    end
end

%% PLOT
disp(r_sum)

% % ALL PLOT ========================================
% figure(1)
% tiledlayout(2,1);
% nexttile
% for j = 1:1:case_num
%     plot(traj_list((1) + 2*(j-1), :))
%     hold on
% end
% title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
% grid on
% 
% % figure(2)
% nexttile
% for j = 1:1:case_num
%     plot(u_list(j, :))
%     hold on
% end
% title("input", 'fontsize',11,'fontname', 'Times New Roman')
% lgd = legend(data_legend, ...
%     'fontsize',11,'fontname', 'Times New Roman');
% lgd.Layout.Tile = 'south';
% lgd.NumColumns = 3;
% grid on
% % sgtitle(plot_names(1));

% SELECTED0 PLOT ====================================
figure(1)
tiledlayout(2,1);
nexttile
for j = selected0
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected0
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected0), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(2));

% SELECTED1 PLOT ====================================
figure(2)
tiledlayout(2,1);
nexttile
for j = selected1
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected1
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected1), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(2));

% SELECTED2 PLOT ====================================
figure(3)
tiledlayout(2,1);
nexttile
for j = selected2
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected2
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected2), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(3));

% SELECTED3 PLOT ====================================
figure(4)
tiledlayout(2,1);
nexttile
for j = selected3
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected3
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected3), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(4));

%% SAVE PLOTS IN CERTAIN FORMAT
if SAVE_PLOT
%     for j = 1:1:length(plot_names)
%         saveas(figure(j), "fig/" + plot_names(j) + ".fig")
%     end

    for j = 1:1:length(plot_names)
        plt = figure(j);
        exportgraphics(plt, "fig/" + plot_names(j) +'.eps')
    end  
end

%% LOCAL FUNCTIONS
function x = step(x, u)

    th = x(1);
    th_dot = x(2);

    g = 10.0;
    dt = 0.05;
    m = 1.0;
    l = 1.0;


    th_dot = th_dot + (3*g/2/l * sin(th) + 3*u/m/l^2) * dt;
    th = th + th_dot*dt;

    x = [th;th_dot];
end

function [A,B,Q,R] = pen()
    g = 10.0;
    dt = 0.05;
    m = 1.0;
    l = 1.0;

    A = [
        1               dt
        3*g/2/l*dt      1
    ];
    B = [0; 3*dt/m/l^2];

    Q = diag([1, 0.1]);
    R = diag(1e-3);
end

function [A,B,nominal] = adapPen(x, u)
    th = x(1);
    th_dot = x(2);

    g = 10.0;
    dt = 0.05;
    m = 1.0;
    l = 1.0;

    A = [
        1                   dt
        3*g/2/l*dt*cos(th)  1
    ];
    B = [0; 3*dt/m/l^2];
    
    Dx = [
        th + th_dot * dt;
        th_dot + (3*g/2/l * sin(th) + 3*u/m/l^2) * dt;
    ];

    nominal.X = x;
    nominal.U = u;
    nominal.DX = Dx - x;
    nominal.Y = x;
end

function u = select_action(x, nn)
    x = [cos(x(1)); sin(x(1)); x(2)];
    u = predict(nn, dlarray(x', "BC"));
%     u = find(u == max(extractdata(u)));
    [u, u_index] = max(u);

    action_list = (-20:1:20) / 10;
    u = action_list(u_index);
end

