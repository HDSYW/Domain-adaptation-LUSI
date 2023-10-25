function [PredictY,W] = TCALSSVM(train_data,test_data,param)
    energy_ratio = 0.999;
    % ---------- Kernel ----------
    X_c_s = train_data.train_features'; %训练数据集特征
    % X_c_u can be sampled test_features
    X_c_u = test_data.test_features'; %测试数据集特征
%     kparam = struct();
%     kparam.kernel_type = 'gaussian';
    K_c_ss=KerF(X_c_s,param,X_c_s);
    K_c_su=KerF(X_c_s,param,X_c_u);
    K_c_uu=KerF(X_c_u,param,X_c_u);
    K_c_tu=KerF(test_data.test_features',param,X_c_u);
    K_c_ts=KerF(test_data.test_features',param,X_c_s);
%     [K_c_ss,kernel_param] = getKernel(X_c_s', kparam);
%     K_c_su = getKernel(X_c_s', X_c_u', kernel_param);
%     K_c_uu = getKernel(X_c_u', kernel_param);
%     K_c_tu = getKernel(test_data.test_features, X_c_u', kernel_param);
%     K_c_ts = getKernel(test_data.test_features, X_c_s', kernel_param);
    tmp_K = [K_c_ss, K_c_su; K_c_ts, K_c_tu];
    
    [nt,ns] = size(K_c_ts);
    
    % ---------- main algorithm ----------
%     fprintf('performing TCA....\n');
    [W,eig_val] = train_sstca(K_c_ss, K_c_su, K_c_uu, X_c_s, X_c_u, train_data.train_labels, param.gamma, param.lambda, param.mu);
    
    ratio = cumsum(eig_val) / sum(eig_val);
    ind = find(ratio > energy_ratio);
%     fprintf('dimension %d saves %f energy....\n', ind(1),ratio(ind(1)));
    W  = W(:,1:ind(1));
%     W  = W(:,1:ind(1));
    K_tilde = tmp_K*(W*W')*tmp_K';
    Kernel = K_tilde(1:ns,1:ns);
    test_kernel = K_tilde(ns+1:ns+nt,1:ns);
    train_kernel= [(1:size(Kernel, 1))' Kernel];
    
    Trn.X=train_kernel(:,2:end);
    Trn.Y=train_data.train_labels;
    param.kpar.ktype='lin';
    param.kpar.kp1=param.kp1;
    [PredictY , ~] = LSSVM(test_kernel , Trn , param); 
    
%     para=sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.p1,0,4,1);
%     model  = svmtrain(train_data.train_labels, train_kernel, para);
%     % ---------- Predict ----------
%         [mv,~] = size(test_kernel);       emv = ones(mv,1);
%         [PredictY,~,decision_values]=svmpredict( emv , test_kernel , model, '-q');
end

