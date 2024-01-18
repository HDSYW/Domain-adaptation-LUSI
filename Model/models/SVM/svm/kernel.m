function K = kernel(X1,X2,kerneltype,can)
% kerneltype 核方法?can 参数
% 'linear'：线性核
% 'gaussian'：高斯核
% 'mullinear':多项式核
switch kerneltype
    case 'linear' % 线性内积 K(v1,v2) = <v1,v2>
        K = X1'*X2;
    case 'gaussian'% K(v1,v2)=exp(-gama||v1-v2||^2)
        delta = can;
        XX = sum(X1'.*X1',2);
        YY = sum(X2'.*X2',2);
        XY = X1'*X2;
        K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
        K = exp(-K./delta);
% K = exp(-delta * (pdist2(	X1',X2').^2)  );  
    case 'mullinear' % K<v1,v2> = (<v1,v2>+c) d;d为多项式的次数，c为多项式额位移常数?
        K = (X1'*X2).^can;
end
end



