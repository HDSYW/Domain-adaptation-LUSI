function [result] = SVM_rbf(Data,K,ktype,pa)
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
    Para.p1 = 2.^j;
    for power= pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;
        for i = 1:k
            test = (indices == i); train = ~test;
            Trn.X = X_train_T(train,:); Trn.Y = Y_train_T(train,:);
            ValX = X_train_T(test,:)  ; ValY = Y_train_T(test,:);
            % ---------- Model ----------
            [PredictY , model] = LIB_L1SVC(ValX , Trn , Para);
            M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
            PredictY(PredictY==0)=-1;ValY(ValY==0)=-1;
            CM = ConfusionMatrix(PredictY,ValY);
            M_F(i)=CM.FM;
            M_GM(i)=CM.GM;
        end
        mean_Acc =mean(M_Acc);  mean_F=mean(M_F); mean_GM=mean(M_GM);
        if  mean_Acc>test_ac    %mean_F>test_Fmean_Acc>test_ac
            test_GM=mean_GM ;  test_F=mean_F                  ;   test_ac=mean_Acc;
            best_p1=Para.p1       ;  best_kp1=Para.kpar.kp1 ;
        end
    end
    fprintf('Complete %s\t\n',num2str((j+8)*100/16))
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T; Trn.Y=Y_train_T;
Para.p1=best_p1; Para.kpar.kp1=best_kp1;
% ---------- Model ----------
[PredictY , ~] = LIB_L1SVC(X_test_T , Trn , Para);
% PredictY(PredictY==0)=-1   ;Y_test_T(Y_test_T==0)=-1;
result.ac_test=sum(PredictY==Y_test_T)/length(Y_test_T)*100;
CM = ConfusionMatrix(PredictY,Y_test_T);
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY, '1');
result.AUC=100*AUC;
result.lam=best_p1;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))        ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)        ; fprintf('BestC=%.2f||',log2(best_p1))   ;
fprintf('Best_kp1=%.2f||\n',log2(best_kp1)) ; fprintf('%s\n', repmat('=', 1, 100));
end

