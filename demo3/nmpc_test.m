clear
%% problem define

x_ref = [];
x0 = [60/3.6 0 0]';
u0 = [0 0 0]';

x_cstr = [
    50/3.6          70/3.6
    -5/3.6          5/3.6
    -5*pi/180       5*pi/180
];

u_cstr = [
    -pi/6           +pi/6 
    -1e3            1e3
    -1e3            1e3
];
%% nmpc parameters
state_num = 3;
input_num = 3;

Ts = 0.01;
Np = 20;
Nc = 10;

modelStataFun = "stateFun";
jacobianStateFun = "stateJacobianFun";
costFun = "optFun";
%% nonlinear model predictive controller define
nlobj = nlmpc(state_num, state_num, input_num);

nlobj.Ts = Ts;
nlobj.PredictionHorizon = Np;
nlobj.ControlHorizon = Nc;

nlobj.Model.StateFcn = modelStataFun;
nlobj.Model.OutputFcn = @(x,u) x;
nlobj.Jacobian.StateFcn = jacobianStateFun;

% nlobj.Optimization.CustomCostFcn = costFun;
% nlobj.Optimization.Re

for j = 1:1:length(x0)
    nlobj.OV(j).Min = x_cstr(j, 1);
    nlobj.OV(j).Max = x_cstr(j, 2);

    nlobj.MV(j).Min = u_cstr(j, 1);
    nlobj.MV(j).Max = u_cstr(j, 2);
end

%% controller validation
validateFcns(nlobj, x0, u0);

%%
[~, ~, info] = nlmpcmove(nlobj, x0, u0);












