clear
close all

%%
max_step = 100;


x0 = [0.3;0];
traj = x0;
u_list = [];


dt = 0.05;


%%
[A,B,Q,R] = pen();
sys = ss(A,B,eye(2),zeros(2,1),dt);
sys = d2c(sys);
[K,S,P] = lqr(sys,Q,R);

x = x0;
for j = 1:1:max_step
    u = -K*x;
    x = step(x, u);

    traj = [traj x];
    u_list = [u_list u];
end



figure(1)
plot(traj(1,:));
hold on
plot(traj(2,:));

figure(2)
plot(cos(traj(1,:)))
hold on
plot(sin(traj(1,:)))
legend

figure(3)
plot(u_list)


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
