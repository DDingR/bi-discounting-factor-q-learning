function [X C I out] = pendulum(inp,par)

g = 10.0;
dt = 0.05;
m = 1.0;
l = 1.0;

Q = diag([1,0.1]);
R = diag(1e-3);

u = inp.U{1};
X{2} = inp.X{2} + (3*g/2/l * sin(inp.X{1}) + 3*u/m/l^2) * dt;
%X{2} = inp.X{2} + (g/l * sin(inp.X{1}) + u/m/l^2) * dt;
%X{1} = inp.X{1} + inp.X{2}*dt;
X{1} = inp.X{1} + X{2}*dt;

% Calculate infeasibility
in_x2 = (abs(X{2}) > 8);

% COST
% Summarize infeasi_batle matrix
I = (in_x2 ~= 0);
% Calculate cost matrix (fuel mass flow)
C{1}  = X{1}.^2*Q(1,1) + X{2}.^2*Q(2,2) + u.^2*R;

if numel(find(I==0))==0
    keyboard
end

% SIGNALS
% store relevant signals in out
out.x1 = X{1};
out.x2 = X{2};
out.u = u;

% REVISION HISTORY
% =========================================================================
% DATE      WHO                 WHAT
% -------------------------------------------------------------------------
% 
