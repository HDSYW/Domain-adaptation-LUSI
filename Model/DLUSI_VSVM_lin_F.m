function [result] = DLUSI_VSVM_lin_F(Data,K,ktype,v_ker,CDFx,p,pa)
test_ac=0;
ptype=p;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
data_ori.X_train_A=Data.X_train_A;
data_ori.Y_train_A=Data.Y_train_A;
k=K;
Y_train_T(Y_train_T==0)=-1;Y_test_T(Y_test_T==0)=-1;
Para.kpar.ktype = ktype;

V_Matrix = 'Vmatrix';  
V_MatrixFun = str2func(V_Matrix);    
Para.vmatrix = V_Matrix;
Para.CDFx = CDFx;  
Para.v_ker = v_ker;    
CDFx = Para.CDFx;

indices = crossvalind('Kfold',data_ori.X_train_A(:,1),k);

% >>>>>>>>>>>>>>>>>>>>Setp 1<<<<<<<<<<<<<<<<<<<<
fprintf('Step 1 %s START !!\n',"*SVM*")
fprintf('%s\n', repmat('-', 1, 100));
for j = pa.min:pa.step:pa.max
    Para.p1 = 2.^j;
    for power=0
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;
        for i = 1:k
            test = (indices == i); Train = ~test;
            TrnX = data_ori.X_train_A(Train,:); TrnY = data_ori.Y_train_A(Train,:);
            ValX = data_ori.X_train_A(test,:)  ; ValY = data_ori.Y_train_A(test,:);
            % ---------- Model ----------
            TrnX = sparse(TrnX);
            ValX = sparse(ValX);
            option = sprintf('-s 3 -B 1 -c %f -q', Para.p1);
            model=train(TrnY, TrnX, option);
            [PredictY, accuracy, ~] = predict(ValY, ValX, model);
            %                                     [PredictY , ~] = LIB_L1SVC(ValX , Trn , Para);
            M_Acc(i) = accuracy(1);
            PredictY(PredictY==0)=-1;ValY(ValY==0)=-1;
            CM = ConfusionMatrix(PredictY,ValY);
            M_F(i)=CM.FM;   M_GM(i)=CM.GM;
        end
        mean_Acc =mean(M_Acc);  mean_F=mean(M_F); mean_GM=mean(M_GM);
        if  mean_Acc>test_ac    %mean_F>test_Fmean_Acc>test_ac
            test_GM=mean_GM ;  test_F=mean_F                  ;   test_ac=mean_Acc;
            best_p1=Para.p1       ;  best_kp1=Para.kpar.kp1 ;
        end
    end
    fprintf('Complete %s\t\n',num2str((j+8)*100/16))
end
% -------------------- Test and prediction --------------------
TrnX=data_ori.X_train_A; TrnY=data_ori.Y_train_A;
TrnX = sparse(TrnX);X_train_T = sparse(X_train_T);
Y_train_T(Y_train_T==-1)=0;
Para.p1=best_p1; Para.kpar.kp1=best_kp1;
% ---------- P ----------
option = sprintf('-s 3 -B 1 -c %f -q', Para.p1);
model=train(TrnY, TrnX, option);
[PredictY_T, accuracy, DV] = predict(Y_train_T, X_train_T, model);
KP.kp1=best_kp1;
KP.ktype=Para.kpar.ktype;
% ---------- R ----------
result.ac_test=accuracy(1);
CM = ConfusionMatrix(PredictY_T,Y_train_T);
result.F=CM.FM; result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_train_T, PredictY_T, '1');
result.AUC=100*AUC; result.lam=best_p1;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))                         ;  fprintf('Test_AC=%.2f||',result.ac_test) ;
fprintf('Train_AC=%.2f||\n',test_ac)                      ;  fprintf('Best_p1=%.2f||',log2(best_p1)) ;
fprintf('Best_kp1=%.2f\n',log2(Para.kpar.kp1)) ;  fprintf('%s\n', repmat('-', 1, 100))           ;

% >>>>>>>>>>>>>>>>>>>>Setp 2<<<<<<<<<<<<<<<<<<<<
%                     Vtype= ["V"];
Vtype= ["I"];
fprintf('Step 2 %s *Vtype=%s* *Ptype=%s* START !!\n',"*DLUSI* ",Vtype, ptype)
test_ac=0;
fprintf('%s\n', repmat('-', 1, 100));
indices = crossvalind('Kfold',X_train_T(:,1),k);
for j = pa.min:pa.step:pa.max
    Para.p1=2.^j;
    for power1 = 0
        Para.v_sig = 2.^power1;
        for power2 = 0
            Para.kpar.kp1 =2.^power2; Para.kpar.kp2 = 0;
            for tao=pa.taomin:pa.taostep:pa.taomax
                Para.p3=tao;
                for i = 1:k
                    test = (indices == i); Train = ~test;
                    Trn.X = X_train_T(Train,:); Trn.Y = Y_train_T(Train,:);
                    ValX = X_train_T(test,:); ValY = Y_train_T(test,:);
                    % ---------- V & P ----------
                    if sum(Vtype=='V')
                        [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
                    else
                        Para.V=eye(size(Trn.X,1),size(Trn.X,1));
                    end
                    [Para.P,Para.p3] = DPcaculate(Trn.X, data_ori.X_train_A, PredictY_T(Train,:),data_ori.Y_train_A, DV(Train,:), KP, ptype, Para.p3);
                    % Para.P=Pcaculate(Trn.X , Trn.Y,ptype);
                    % Para.P=PredictY_T(train,:)*PredictY_T(train,:)';
                    % ---------- Model ----------
                    [PredictY , model] = LUSI_VSVM_lin(ValX , Trn , Para);
                    PredictY.tst(PredictY.tst==-1)=0;
                    M_Acc(i) = sum(PredictY.tst==ValY)/length(ValY)*100;
                    CM = ConfusionMatrix(PredictY.tst,ValY);  M_F(i)=CM.FM;
                end
            end
            mean_Acc =mean(M_Acc); mean_F=mean(M_F); 
            if  mean_Acc>test_ac       % mean_Acc>test_ac or mean_F>test_F
                test_ac=mean_Acc       ; 
                best_v_sig=Para.v_sig  ; best_kp1=Para.kpar.kp1; best_p1=Para.p1;best_p3=Para.p3;
            end
        end
    end
    fprintf('Complete %s\t\n',num2str((j+8)*100/16))
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T; Trn.Y=Y_train_T;
Para.v_sig=best_v_sig;   Para.kpar.kp1=best_kp1; Para.p1=best_p1;Para.p3=best_p3;
% ---------- V & P----------
if sum(Vtype=='V')
    [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
else
    Para.V=eye(size(Trn.X,1),size(Trn.X,1));  Para.V=eye(size(Trn.X,1),size(Trn.X,1));
end
[Para.P,Para.p3] = DPcaculate(Trn.X, data_ori.X_train_A, PredictY_T,data_ori.Y_train_A,DV, KP, ptype, Para.p3);
% ---------- Model ----------
[PredictY , ~] = LUSI_VSVM_lin(X_test_T , Trn , Para);
CM = ConfusionMatrix(PredictY.tst,Y_test_T) ;
result.ac_test=sum(PredictY.tst==Y_test_T)/length(Y_test_T)*100;
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY.tst, '1');
result.AUC=100*AUC    ;  result.lam=best_p1    ;  result.kp1=best_kp1;
result.v_sig=best_v_sig ; result.tao_1=best_p3 ;  result.Ptype=ptype;
fprintf('%s\n', repmat('-', 1, 100))        ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)     ; fprintf('Best_v_sig=%.4f||',log2(best_v_sig));
fprintf('BestC=%.2f||',log2(best_p1))  ; fprintf('Best_kp1=%.2f||',log2(best_kp1));
fprintf('Best_tao=%.2f||\n',best_p3);  fprintf('%s\n', repmat('=', 1, 100));
end

