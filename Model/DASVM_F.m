function [result] = DASVM_F(Data,K,ktype,pa)
test_ac=0;
k=K;
Para.kpar.ktype = ktype;
X=Data.X_train_A;
Y=Data.Y_train_A;
Y(Y==0)=-1;
data_ori.X_test=Data.X_test;
data_ori.Y_test=Data.Y_test;
data_ori.Y_test(data_ori.Y_test==0)=-1;
indices = crossvalind('Kfold',X(:,1),k);

Para.rho  = 10;
Para.max_iter  = 100;
for max_unl_num=3
    Para.max_unl_num = 2.^max_unl_num;
    for j = pa.min:pa.step:pa.max
        Para.p1=2.^j;
        Para.Cu = Para.p1*0.1;% Cu should be less than C
        Para.Cu_max  = 0.5*Para.p1;
        for power2 = pa.min:pa.step:pa.max
            Para.kpar.kp1 =2.^power2;
            Para.kpar.kp2 = 0;
            for i = 1:k
                test = (indices == i); train = ~test;
                Trn.X = X(train,:); Trn.Y = Y(train,:);
                Trn.XY=[Trn.X,Trn.Y];
                pos_features = Trn.XY(Trn.XY(:,end)==1,1:end-1);
                neg_features = Trn.XY(Trn.XY(:,end)==-1,1:end-1);
                ValX = data_ori.X_test  ; ValY = data_ori.Y_test;
                % ---------- Model ----------
                [PredictY,model] = DASVM(pos_features, neg_features, ValX, Para);
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
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X ;  Trn.Y=Y;
pos_features = Trn.XY(Trn.XY(:,end)==1,1:end-1);
neg_features = Trn.XY(Trn.XY(:,end)==-1,1:end-1);
Para.kpar.kp1=best_kp1; Para.p1=best_p1;
% ---------- Model ----------
[PredictY,~] = DASVM(pos_features, neg_features, data_ori.X_test, Para);
CM = ConfusionMatrix(PredictY,data_ori.Y_test) ;
result.ac_test=sum(PredictY==data_ori.Y_test)/length(data_ori.Y_test)*100;
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(data_ori.Y_test, PredictY, '1');
result.AUC=100*AUC    ;  result.lam=best_p1    ;  result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))        ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)     ; fprintf('BestC=%.2f||',log2(best_p1))  ;
fprintf('Best_kp1=%.2f||\n',log2(best_kp1)); fprintf('%s\n', repmat('=', 1, 100));
end

