function [V,v] = Vmatrix(X,CDFx,Sigma,v_ker)
% _______________________________ Input  _______________________________
%      X  -  m x n matrix, explanatory variables in training data
%      CDFx  -  mu(x) in the V-matrix expression, including:  'uniform', 'normal','empirical'
%       v_ker -  kernel type in V-matrix, including: 'theta_ker', 'gaussian_ker'
%      Sigma  -  v-kernel para of 'gussian_ker'
% ______________________________ Output  ______________________________
%     V-  V-matrix
%     v - A cell for each dimensional v-matrix
%     Lateset update: 2022.07.12.


% X = X(1:300,:);
% CDFx = 'uniform';
% v_ker = 'gaussian_ker';
% Sigma = 2^(-6);

[nsamp,nfea] = size(X);
% X=sparse(X);
%=======Uniform Distribution=========
switch CDFx
    case 'uniform'
        V = zeros(nsamp,nsamp);
        for d = 1:nfea
            [a,b,~,~] = unifit(X(:,d),0.02);
            if a==b
                v{d} = zeros(nsamp);
            else
                switch v_ker
                    case 'theta_ker'
                        Xd = X(:,d);
                        XI=repmat(Xd,1,nsamp);
                        XJ= XI';
                        M = max(XI,XJ);
                        v{d} = (b-M)/(b-a);
                    case 'gaussian_ker'
                        Xd = X(:,d);
                        XX = sum(Xd.*Xd,2);
                        K = abs(repmat(XX,[1 size(XX ,1)]) + repmat(XX',[size(XX,1) 1]) - 2*(Xd*Xd'));
                        K = exp(-K./((2*Sigma)^2));
                        G=1/(b-a)*K;
                        B=repmat(Xd,1,nsamp);
                        D=(B+B')/2;
                        GG=erf((D-a)/Sigma)+erf((b-D)/Sigma);
                        v{d} = G.*GG;
                        if a==b
                            v{d}= zeros(nsamp,nsamp);
                        end
                end       
            end
            V = V + v{d};
        end
        %=============Normal Distribution==========
    case 'normal'
        alpha = 0.05;
        V = ones(nsamp,nsamp);
        % X = zscore(X);
        for d = 1:nfea
            [mu,sigma,~,~] = normfit(X(:,d),alpha);
            switch v_ker
                case 'theta_ker'
                    Xd = X(:,d);
                    XI=repmat(Xd,1,nsamp);
                    XJ= XI';
                    M = max(XI,XJ);
                    cdf{d} = normcdf(M,mu,sigma);
                    v{d} = 1-cdf{d};
                case 'gaussian_ker'
            end
            V = V + v{d};
        end
    case 'empirical'
end
end





