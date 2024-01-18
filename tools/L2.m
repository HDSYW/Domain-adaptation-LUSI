function  [P]=L2(X,Y)
m=length( Y);
zheng=X(Y==1,:);
[n,~]=size(zheng);
S=sum(zheng,1)/n;% 正类特征的平均值
SS=repmat(S,m, 1);
% SS=S';
G=X-SS;%所有样本减去正类样本的平均值
G=sum(abs(G),2);
P=G*G';
% Y= repmat(Y,1,m); 
Y=Y*Y';
P=Y.*P;
end