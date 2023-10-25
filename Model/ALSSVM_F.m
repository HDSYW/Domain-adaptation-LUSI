function [result] = ALSSVM_F(Data,K,ktype,pa)
test_ac=0;
k=K;
Para.kpar.ktype = ktype;
data_ori.X_train_A=Data.X_train_A;
data_ori.Y_train_A=Data.Y_train_A;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
data_ori.Y_train_A(data_ori.Y_train_A==0)=-1;
Y_test_T(Y_test_T==0)=-1;
Y_train_T(Y_train_T==0)=-1;

% >>>>>>>>>>>>>>>>>>>>Setp 1<<<<<<<<<<<<<<<<<<<<
fprintf('Step 1 %s START !!\n',"*SVM^a*")
fprintf('%s\n', repmat('-', 1, 100));
indices = crossvalind('Kfold',data_ori.X_train_A(:,1),k);
for p1= pa.min:pa.step:pa.max
    Para.p1 = 2.^p1;
    for power= pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;
        for i = 1:k
            test = (indices == i); train = ~test;
            Trn.X = data_ori.X_train_A(train,:); Trn.Y = data_ori.Y_train_A(train,:);
            ValX = data_ori.X_train_A(test,:)  ;  ValY = data_ori.Y_train_A(test,:);
            % ---------- Model ----------
            [PredictY , ~] = LIB_L1SVC(ValX , Trn , Para);
            M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
        end
        mean_Acc =mean(M_Acc);
        if  mean_Acc>test_ac
            test_ac=mean_Acc;  best_p1=Para.p1 ;  best_kp1=Para.kpar.kp1 ;
        end
    end
    fprintf('Complete %s\t\n',num2str((p1+8)*100/16))
end
% -------------------- Test and prediction --------------------
Trn.X=data_ori.X_train_A; Trn.Y=data_ori.Y_train_A;
Para.p1=best_p1; Para.kpar.kp1=best_kp1;
% ---------- P ----------
[PredictY_T , model_train] = LIB_L1SVC(X_train_T , Trn , Para);
DV=model_train.dv;
[PredictY_T_L , model_train_L] = LIB_L1SVC(X_test_T , Trn , Para);
DV_L=model_train_L.dv;
KP.kp1=best_kp1;
KP.ktype=Para.kpar.ktype;
% ---------- R ----------
result.ac_test=sum(PredictY_T==Y_train_T)/length(Y_train_T)*100;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))           ;  fprintf('Test_AC=%.2f||',result.ac_test) ;
fprintf('Train_AC=%.2f||\n',test_ac)           ;  fprintf('Best_p1=%.2f||',log2(best_p1)) ;
fprintf('Best_kp1=%.2f\n',log2(Para.kpar.kp1)) ;  fprintf('%s\n', repmat('-', 1, 100))           ;

% >>>>>>>>>>>>>>>>>>>>Setp 2<<<<<<<<<<<<<<<<<<<<
fprintf('Step 2 %s START !!\n',"*LSSVM*")
fprintf('%s\n', repmat('-', 1, 100));
test_ac=0;
kernel1=pa.kernel1;
indices = crossvalind('Kfold',X_train_T(:,1),k);
for p1= pa.min:pa.step:pa.max
    Para.p1=2.^(p1);
    for power= pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;
        for j = 1:k
            test = (indices == j); train = ~test;
            DataTrain.X = X_train_T(train,:) ; DataTrain.Y = Y_train_T(train,:);
            DataTest.X = X_train_T(test,:)    ;  DataTest.Y = Y_train_T(test,:);
            [ACC,pred] = LSSVM_(DataTrain.X ,DataTrain.Y,DataTest.X,DataTest.Y,kernel1,Para.p1,Para.kpar.kp1,DV(train,:),DV(test,:));
%             svm=train_svm(DataTrain.X',DataTrain.Y',kernel1,kc,Para.p1,DV(train,:));
%             [r,~]=test_svm(svm,DataTest.X',DataTest.Y',kernel1,kc,DV(test,:));
            M_Acc(j) =ACC./100;
        end
        mean_Acc =mean(M_Acc);  
        if  mean_Acc>test_ac
            test_ac=mean_Acc;  best_p1=Para.p1;  best_kp1=Para.kpar.kp1;
        end
    end
    fprintf('Complete %s\t\n',num2str((p1+8)*100/16))
end
% -------------------- Test and prediction --------------------
DataTrain.X=X_train_T; DataTrain.Y=Y_train_T;
Para.p1=best_p1; Para.kpar.kp1=best_kp1;
[ACC,PredictY] = LSSVM_(DataTrain.X ,DataTrain.Y,X_test_T,Y_test_T,kernel1,Para.p1,Para.kpar.kp1,DV,DV_L);
% ---------- R ----------
result.ac_test=sum(PredictY==Y_test_T)/length(Y_test_T)*100;
CM = ConfusionMatrix(PredictY,Y_test_T);
result.F=CM.FM; result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY, '1');
result.AUC=100*AUC; result.lam=best_p1;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))                         ;  fprintf('Test_AC=%.2f||',result.ac_test) ;
fprintf('Train_AC=%.2f||\n',test_ac*100)                      ;  fprintf('Best_p1=%.2f||',log2(best_p1)) ;
fprintf('Best_kp1=%.2f\n',log2(Para.kpar.kp1)) ;  fprintf('%s\n', repmat('-', 1, 100))           ;
end

