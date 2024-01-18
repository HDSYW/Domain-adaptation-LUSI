function [X_train,Y_train,X_test,Y_test] = TT(X,Y,percent)
perc=percent;
Y(Y==2)=-1;
Y(Y==0)=-1;
test=[X,Y];
%  train=[X_train,Y_train];
idxz=test(:,end)==1;
data_z=test(idxz,:);
idxf=test(:,end)==-1;
data_f=test(idxf,:);
idx_z=randperm(size(data_z,1), floor(perc*size(data_z,1)))';
idx_f=randperm(size(data_f,1), floor(perc*size(data_f,1)))';
D_test=[(data_z(idx_z,:));(data_f(idx_f,:))];
idx1_z=setdiff((1:size(data_z,1))',idx_z,'row','stable');
idx1_f=setdiff((1:size(data_f,1))',idx_f,'row','stable');
D_train = [(data_z(idx1_z,:));(data_f(idx1_f,:))];
X_test=D_test(:,1:end-1);
Y_test=D_test(:,end);
X_train=D_train(:,1:end-1);
Y_train=D_train(:,end);
% X_train=[zz(1:floor(perc*size(zz,1)),1:end-1);ff(1:floor(perc*size(ff,1)),1:end-1)];
% X_test =[zz((floor(perc*size(zz,1))+1):end,1:end-1);ff((floor(perc*size(zz,1))+1):end,1:end-1)];
% Y_train=[zz(1:floor(perc*size(zz,1)),end);ff(1:floor(perc*size(ff,1)),end)];
% Y_test =[zz((floor(perc*size(zz,1))+1):end,end);ff((floor(perc*size(ff,1))+1):end,end)];
end

