function cost = optFun(x,u,e,data)

Q = diag([1, 0.1]);
R = diag(1e-3);

% x_norm = [rem(x(:,1)+pi, 2*pi) - pi; x(:,2)];

x_1 = rem(x(:,1) + pi, 2*pi) - pi;
x_2 = x(:,2);

cost = x_1'*x_1 * Q(1,1) + x_2'*x_2 * Q(2,2) + u'* u * R;
end