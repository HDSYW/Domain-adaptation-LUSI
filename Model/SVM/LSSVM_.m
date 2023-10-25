function [ACC,pred] = LSSVM_(X_train,Y_train,X_test,Y_test,kerneltype,C,B,DV,DV_L)
kc.c1=B;
A=Y_train*Y_train'.*kernel(X_train',X_train',kerneltype,kc);
[m,~]=size(A);
J=eye(m);
J=(1/C)*J;
V=A+J;
BB=Y_train;
CC=[Y_train',0];
GG=[V,BB];
MM=[GG;CC];
[m,~]=size(Y_train);
hh=ones(m,1)-Y_train.*DV;
h=[hh;0];
%  Z=ll*h;
Z=lsqminnorm(MM,h);% Z是一个包含阿尔法的和b的向量

[m,~]=size(Y_train);
[n,~]=size(Y_test);
w=Z(1:m,:);
gb=Z(m+1,:);
kc.c1=B;
b=gb*ones(1,n);
w = (w.* Y_train)'*kernel(X_train',X_test',kerneltype,kc)+b+DV_L';
pred=sign(w)';
ACC=size(find(pred==Y_test))/size(Y_test)*100;%A是准确率
end
