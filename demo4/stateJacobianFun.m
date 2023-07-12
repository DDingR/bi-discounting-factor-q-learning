function [A, B] = stateJacobianFun(x, u)

Ca = 756.349/(0.6*pi/180);
m = 1644.802;
Iz = 2488.893;
lf = 1.240;
lr = 1.510;
w = 0.8;

x_dot = x(1);
y_dot = x(2) ;   
yaw_dot = x(3);
delta = u(1);        
% Frl = u(2);
% Frr = u(3);


A = [
    [                                               -(2*Ca*sin(delta)*(y_dot + lf*yaw_dot))/(m*x_dot^2),             yaw_dot + (2*Ca*sin(delta))/(m*x_dot),                   y_dot + (2*Ca*lf*sin(delta))/(m*x_dot)]
    [((2*Ca*(y_dot - lr*yaw_dot))/x_dot^2 + (2*Ca*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/m - yaw_dot,       -((2*Ca)/x_dot + (2*Ca*cos(delta))/x_dot)/m, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/m - x_dot]
    [  -((2*Ca*lr*(y_dot - lr*yaw_dot))/x_dot^2 - (2*Ca*lf*cos(delta)*(y_dot + lf*yaw_dot))/x_dot^2)/Iz, ((2*Ca*lr)/x_dot - (2*Ca*lf*cos(delta))/x_dot)/Iz,   -((2*Ca*cos(delta)*lf^2)/x_dot + (2*Ca*lr^2)/x_dot)/Iz]
];

B = [ 
    [      -(2*Ca*sin(delta) + 2*Ca*cos(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,   1/m,  1/m]
    [       (2*Ca*cos(delta) - 2*Ca*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/m,     0,    0]
    [(2*Ca*lf*cos(delta) - 2*Ca*lf*sin(delta)*(delta - (y_dot + lf*yaw_dot)/x_dot))/Iz, -w/Iz, w/Iz]
];


end