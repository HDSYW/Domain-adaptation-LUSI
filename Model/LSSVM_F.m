function [result] = LSSVM_F(Data,K,ktype,pa)
test_ac=0;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
k=K;
Para.kpar.ktype = ktype;
Y_train_T(Y_train_T==0)=-1;Y_test_T(Y_test_T==0)=-1;
indices = crossvalind('Kfold',X_train_T(:,1),k);

for j = pa.min:pa.step:pa.max
    Para.p1=2.^j;
    for power2 = pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power2;
        Para.kpar.kp2 = 0;
        for i = 1:k
            test = (indices == i); train = ~test;
            Trn.X = X_train_T(train,:);Trn.Y = Y_train_T(train,:);
            ValX =X_train_T(test,:)  ; ValY = Y_train_T(test,:);
            % ---------- Model ----------
            [PredictY , model] = LSSVM(ValX , Trn , Para);
            M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
            CM = ConfusionMatrix(PredictY,ValY);
            M_F(i)=CM.FM;
        end
        mean_Acc =mean(M_Acc); mean_F=mean(M_F);
        if  mean_Acc>test_ac       % mean_Acc>test_ac or mean_F>test_F
            test_ac=mean_Acc;         test_ac=mean_Acc;     best_kp1=Para.kpar.kp1; best_p1=Para.p1;
        end
    end
    fprintf('Complete %s\t\n',num2str((j+8)*100/16))
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T;   Trn.Y=Y_train_T;
Para.kpar.kp1=best_kp1; Para.p1=best_p1;
% ---------- Model ----------
[PredictY , ~] = LSSVM(X_test_T , Trn , Para);
PredictY(PredictY==0)=-1   ;Y_test_T(Y_test_T==0)=-1;
CM = ConfusionMatrix(PredictY,Y_test_T) ;
result.ac_test=sum(PredictY==Y_test_T)/length(Y_test_T)*100;
result.lam=best_p1;
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY, '1');
result.AUC=100*AUC;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))               ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)            ; fprintf('BestC=%.2f||',log2(best_p1))   ;
fprintf('Best_kp1=%.2f||\n',log2(best_kp1)) ; fprintf('%s\n', repmat('=', 1, 100));
end

