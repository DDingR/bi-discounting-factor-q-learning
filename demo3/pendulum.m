clear
close all

%% USER CONSTANT
max_step = 200;

x0 = [3;0];
u0 = 0;

Np = 100; Nc =Np;

TRAIN_NAME = "test_train13";

NN_NAME = [
    "DP"
%     "LQR"
%     "MPC"    
%     "NLMPC"
     "0_0_0_end"
     "1_0_0_end"
     "2_0_0_end";
     "3_0_0_end";
   "4_0_0_end";
     "5_0_0_end";
     "6_0_0_end";
     "7_0_0_end";
     "8_0_0_end";

    ];

data_legend = [
    "DP"
%     "LQR"
%     "MPC"
%     "NLMPC"
     "eps 0.99"
     "eps 0.75"
     "eps 0.5"
     "eps 0.25"
   "eps 0"    
   "gamma 1.25"    
   "gamma 1.5"    
   "gamma 1.25 eps 0"    
   "gamma 1.5 eps 0"    
%     "gamma 0.99, epsilon 0"    
    ];

selected = [1 2 7];

%% CONSTANTS
case_num = size(NN_NAME, 1);

traj_list = [repmat(x0, [case_num, 1]) zeros(2*case_num, max_step-1)];
u_list = zeros(case_num, max_step);
t_list = zeros(case_num, max_step);
r_sum = zeros(case_num, 1);

dt = 0.01;

%% CONVENTIOANL CONTROL
[A,B,Q,R] = pen();
sysd = ss(A,B,eye(2),zeros(2,1),dt);
sys = d2c(sysd);

% LQR ================================
tic
[K,S,P] = lqr(sys,Q,R);
lqr_t = toc;
% MPC ================================
[A,B,nominal] = adapPen(x0, u0);
sysd = ss(A,B,eye(2),zeros(2,1),dt);
sys = d2c(sysd);

% mpcobj = mpc(sysd, Np, Nc);

mpcobj = mpc(sysd, dt, Np, Nc);
ms = mpcstate(mpcobj);

mpcobj.OutputVariables(2).Min = -8;
mpcobj.OutputVariables(2).Max = +8;
mpcobj.ManipulatedVariables.Min = -2;
mpcobj.ManipulatedVariables.Max = +2;
mpcobj.Weights.ManipulatedVariables = 1e-3;
mpcobj.Weights.OutputVariables = [1 0.1];
mpcobj.Weights.ManipulatedVariablesRate = 0;


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

%% DEMOSTRAION

for j = 1:1:case_num
    x = x0; u = u0;
    if NN_NAME(j) ~= "DP" && NN_NAME(j) ~= "LQR" && NN_NAME(j) ~= "MPC" && NN_NAME(j) ~= "NLMPC"
        NN_PATH = "./onnx/" + TRAIN_NAME + "/" + NN_NAME(j) + ".onnx";
        nn = importONNXNetwork(NN_PATH, TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC");
    end
    r = 0;

    for k = 1:1:max_step
        tic
        if NN_NAME(j) == "DP"
            u = res.u(k);
        elseif NN_NAME(j) == "LQR"
            u = -K*x;
        elseif NN_NAME(j) == "MPC"
            [A,B,nominal] = adapPen(x, u);
            sysd = ss(A,B,eye(2),zeros(2,1),dt);
            sys = d2c(sysd);

%             opt = mpcmoveopt;
%             opt.

            [u, info] = mpcmoveAdaptive(mpcobj, mpcstate(mpcobj), sysd, nominal, x, [0;0]);
%             u = mpcmove(mpcobj, mpcstate(mpcobj), x, [0;0]);
        elseif NN_NAME(j) =="NLMPC"
            u = nlmpcmove(nlobj, x, u);
        else
            u = select_action(x, nn);
        end
        t = toc;
        
        x = step(x, u);
       
	r = x'*Q*x + u'*R*u;

        traj_list((1:2) + 2*(j-1), k) = x;
        u_list(j, k) = u;
        t_list(j, k) = t;
	r_sum(j, 1) = r_sum(j, 1) + r;
    end
end

%% PLOT
disp(r_sum)

figure(1)
subplot(2,1,1)
for j = 1:1:case_num
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
legend(data_legend, "location", "southeast")

% figure(2)
subplot(2,1,2)
for j = 1:1:case_num
    plot(u_list(j, :))
    hold on
end
legend(data_legend, "location", "southeast")

% figure(3)
% for j = selected
%     plot(traj_list((1) + 2*(j-1), :))
%     hold on
% end
% legend(data_legend(selected), "location", "southeast")
% 
% figure(4)
% for j = selected
%     plot(u_list(j, :))
%     hold on
% end
% legend(data_legend(selected), "location", "southeast")

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
    dt = 0.01;
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
    dt = 0.01;
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
