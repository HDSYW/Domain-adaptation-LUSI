function [PredY,model] = KMM_LSSVM(TestX,DataTrain,Para)
    B=1000;
    n=size(DataTrain.X,1);
    p=size(TestX,1);
    eps=(sqrt(n+p)-1)/sqrt(n+p);
    options=optimoptions('quadprog','Display','off');
    H=KerF(DataTrain.X,Para.kpar,DataTrain.X);
    f=-n/p*sum(KerF(DataTrain.X,Para.kpar,TestX),2);
    A=[ones(1,n);-ones(1,n)];
    bb=[n*(eps+1),n*(eps+1)];
    lb=[zeros(n,1)];
    ub=[B*ones(n,1)];
    beta =quadprog(H,f,A,bb,[],[],lb,ub,[],options);
    
    gam = Para.p1;      sig2 = Para.kpar.kp1;
    kpar = Para.kpar;
    X = beta.*beta.*KerF(DataTrain.X,Para.kpar,DataTrain.X);       
%     X = KerF(DataTrain.X,Para.kpar,DataTrain.X);       
    Y = DataTrain.Y;    
    TestX=KerF(TestX,Para.kpar,DataTrain.X);
    clear DataTrain
%     if strcmp(kpar.ktype,'lin')
%         ker = 'lin_kernel';
%     elseif strcmp(kpar.ktype,'rbf')
        ker = 'RBF_kernel';
%     end
%      ker = 'lin_kernel';
    
    t = tic;
    [alph,b] = trainlssvm( {X,Y,'c',gam,sig2,ker} );
    trn_time = toc(t);
    
    [PredY,~]= simlssvm( {X,Y,'c',gam,sig2,ker} , {alph,b} , TestX);
%     model.beta=beta;
    model.alph = alph;
    model.b = b;
    model.n_SV = length(alph);
    model.ind_SV = find(abs(alph) > 1e-6);
    model.tr_time = trn_time;
%     if Para.drw == 1
%         drw.ds = dec_val;
%         drw.ss1 = drw.ds - 1;
%         drw.ss2 = drw.ds + 1;
%         model.drw = drw;
%         model.twin = 0;
%     end
end






% Full syntax
%
%     1. Using the functional interface:
%
% >> [Yt, Zt] = simlssvm({X,Y,type,gam,sig2}, Xt)
% >> [Yt, Zt] = simlssvm({X,Y,type,gam,sig2,kernel}, Xt)
% >> [Yt, Zt] = simlssvm({X,Y,type,gam,sig2,kernel,preprocess}, Xt)
% >> [Yt, Zt] = simlssvm({X,Y,type,gam,sig2,kernel,preprocess}, {alpha,b}, Xt)
%
%       Outputs
%         Yt            : Nt x m matrix with predicted output of test data
%         Zt(*)         : Nt x m matrix with predicted latent variables of a classifier
%       Inputs
%         X             : N x d matrix with the inputs of the training data
%         Y             : N x 1 vector with the outputs of the training data
%         type          : 'function estimation' ('f') or 'classifier' ('c')
%         gam           : Regularization parameter
%         sig2          : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%         kernel(*)     : Kernel type (by default 'RBF_kernel')
%         preprocess(*) : 'preprocess'(*) or 'original'
%         alpha(*)      : Support values obtained from training
%         b(*)          : Bias term obtained from training
%         Xt            : Nt x d inputs of the test data
%
%
%     2. Using the object oriented interface:
%
% >> [Yt, Zt, model] = simlssvm(model, Xt)
%
%       Outputs
%         Yt       : Nt x m matrix with predicted output of test data
%         Zt(*)    : Nt x m matrix with predicted latent variables of a classifier
%         model(*) : Object oriented representation of the LS-SVM model
%       Inputs
%         model    : Object oriented representation of the LS-SVM model
%         Xt       : Nt x d matrix with the inputs of the test data