function svm = train_svm(train_data,train_label,kertype,kc,C,Y_A)%  里边的输入不可以带点
n = length(train_label); % 计算样本个数
can = kc;
H = (train_label'*train_label).*kernel(train_data,train_data,kertype,can);
lambda=train_label'.*Y_A;
f = -(ones(n,1)-lambda); % 保证f为列向量，原式中包含转置的操作
A = [];% 不含不等约束
b = [];% 不含不等约束
Aeq = train_label;beq = 0;% s.t.: aY = 0;
lb = zeros(n,1);
ub = C*ones(n,1);                       %C*ones(n,1); % 0 <= a <= C
a0 = [];    % 对解进行初始化?
options = optimoptions('quadprog','Display','off');
a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
a(a<1e-6)=0;a(C-a<1e-6)=C;
%e = 1e-10;
sv_index = find(a>0);
svm.a = a(sv_index);
svm.data = train_data(:,sv_index);
svm.label = train_label(:,sv_index);
end