function  [P]=L1(X,Y)%正类 
m=length( Y);
zheng=X(Y==1,:);%去出正类样本
[n,~]=size(zheng);
S=sum(zheng,1)/n;% 计算正类样本的 每一维平均值
% SS=repmat(S,m, 1);
SS=S';
    P=dist(X,SS);%矩阵X和矩阵SS之间的欧式距离: 其中,X的列数,应与SS的行数相等。 计算时,计算的是X每一行和SS的每一列的欧式距离。
    P=P*P';%数据与正类样本均值的距理
Y= repmat(Y,1,m); 
Y=Y.*Y';
P=Y.*P;
end