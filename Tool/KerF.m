function K = KerF(TstX, kpar, X)
% Construct the positive (semi-) definite (symmetric) kernel matrix
%
% Inputs: 
%     TstX          - mt x n matrix, #m1 Test vector with n dimension
%     kpar.type - kernel type (by default 'RBF_kernel')
%     kpar.kp1 - kernel parameter 1 
%     kpar.kp2 - kernel parameter 2 
%     X                - m x n matrix, #m BASE vector with n dimension
% Outputs: 
%     K                -  mt x m kernel matrix
%     K(a,b)        -  the similarity measure of vector TstX(a) and X(b)
% 
% Written by Lingwei Huang, lateset update: 2021.09.15. 
% Copyright 2019-2021  Lingwei Huang. 

%% Input 
    ktype = kpar.ktype;
    kp1 = kpar.kp1;
%     if length(unique(fieldnames(kpar))) <= 1
%         kp2 = 0;
%     else
%         kp2 = kpar.kp2;
%     end

%     [m,n] = size(X);
%     [mt,~] = size(TstX);
    
%% Compute Kernel

    if strcmp(ktype,'lin') % linear 线性核
        K = TstX * X';
    end


    if strcmp(ktype,'poly') % polynomial 多项式核
        gamma = 1;       % default 1 
        coef0 = kp2;    % default 1 
        degree = kp1; % default 3 
        K = ( gamma * TstX*X' + coef0 ) .^ degree;
    end


    if strcmp(ktype,'rbf') % radial basis function 径向基核
        gamma = kp1; % gamma=1/(2*sig^2), default 1/n 
        K = exp(  -gamma .* (pdist2(TstX,X).^2)  );     % lib version
%         sig = kp1;
%         K = exp(  - (pdist2(TstX,X).^2) ./ (2*sig^2)  ); % micro version
%         sig2 = kp1;
%         K = exp(  - (pdist2(TstX,X).^2) ./ (2*sig2)  );    % old version
    end


    if strcmp(ktype,'rbfnt') % rbf with negative tail
        gamma = kp1; % gamma=1/(2*sig^2), default 1/n 
        coef0 = kp2;    % default 1 
        K = (  coef0 - gamma .* (pdist2(TstX,X).^2)  ) * ...
                   exp(  -gamma .* (pdist2(TstX,X).^2)  ); 
    end


    if strcmp(ktype,'sig')
        gamma = kp1; % default 1/n 
        coef0 = kp2;    % default 0 
        K = tanh( gamma * TstX*X' + coef0 );
    end


%     if strcmp(ktype,'sinc') % 正弦核
%         emt = ones(mt,1); 
%         em = ones(m,1); 
%         XXh1 = TstX*emt * em';
%         XXh2 =      X*em   * emt';
%         K = XXh1-XXh2';
%         K = sinc(kp1.*K);
%     end


%     if strcmp(ktype,'wav') % 小波核
%         XXh1 = sum(TstX.^2,2)*ones(1,m);
%         XXh2 = sum(X.^2,2)*ones(1,mt);
%         K = XXh1+XXh2' - 2*(TstX*X');
%         
%         XXh11 = sum(TstX,2)*ones(1,m);
%         XXh22 = sum(Xt,2)*ones(1,mt);
%         K1 = XXh11-XXh22';
%         
%         K = cos(kp2.*K1).*exp(-kp1.*K);
%     end


end % end KerF



