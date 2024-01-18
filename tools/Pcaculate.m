function P = Pcaculate(X,Y,ptype)
Ptype = num2str(ptype);
Y(Y==-1,1)=0;%%
switch  Ptype
    case '1'
        P =diag(Y)*diag(Y)';
    case '2'
        P =eye(length(Y))*(eye(length(Y)))';
    case '3'
        P =X*X';
    case '4'
        r_p = X*pca(X);%通过某种线性投影，将高维的数据映射到低维的空间中，并期望在所投影的维度上数据的信息量最大（方差最大），以此使用较少的数据维度，同时保留住较多的原数据点的特性。
        P = r_p(:,1)*r_p(:,1)';
    case  '5'
        [P]=L1(X,Y);
    case  '6'
        [P]=L2(X,Y);
    case  '7'
        [P]=L3(X,Y);
    case'8'
        D1 = X(:,1);
        P = D1*D1';
    case'9'
        P =   ones(length(Y),1)*ones(length(Y),1)';
    case'10'
        P =   Y*Y';
end
    end