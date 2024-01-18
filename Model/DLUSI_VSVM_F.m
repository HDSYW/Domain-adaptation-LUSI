function [result] = DLUSI_VSVM_F(Data,K,ktype,v_ker,CDFx,p,pa)

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
    for power=pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;
        for i = 1:k
            test = (indices == i); train = ~test;
            Trn.X = data_ori.X_train_A(train,:); Trn.Y = data_ori.Y_train_A(train,:);
            ValX = data_ori.X_train_A(test,:)  ;  ValY = data_ori.Y_train_A(test,:);
            % ---------- Model ----------
            [PredictY , ~] = LIB_L1SVC(ValX , Trn , Para);
            M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
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
Trn.X=data_ori.X_train_A; Trn.Y=data_ori.Y_train_A;
Para.p1=best_p1; Para.kpar.kp1=best_kp1;
% ---------- P ----------
[PredictY_T , model_train] = LIB_L1SVC(X_train_T , Trn , Para);
DV=model_train.dv;
KP.kp1=best_kp1;
KP.ktype=Para.kpar.ktype;
% ---------- R ----------
Y_train_T(Y_train_T==-1)=0;
result.ac_test=sum(PredictY_T==Y_train_T)/length(Y_train_T)*100;
result.lam=best_p1;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))                         ;  fprintf('Test_AC=%.2f||',result.ac_test) ;
fprintf('Train_AC=%.2f||\n',test_ac)                      ;  fprintf('Best_p1=%.2f||',log2(best_p1)) ;
fprintf('Best_kp1=%.2f\n',log2(Para.kpar.kp1)) ;  fprintf('%s\n', repmat('-', 1, 100))           ;
% >>>>>>>>>>>>>>>>>>>>Setp 2-1<<<<<<<<<<<<<<<<<<<<
Vtype= pa.Vtype;
fprintf('Step 2 *Vtype=%s* START !!\n',Vtype)
test_ac=0;
fprintf('%s\n', repmat('-', 1, 100));
indices = crossvalind('Kfold',X_train_T(:,1),k);
PT_old=["A_Y","DV","Kernel","Spearman1","Kendall1","pearson1","Spearman2","Kendall2","pearson2"];
% for num=1:9
%     pt=PT_old(num);t=0;
%     for power2 = pa.min:pa.step:pa.max
%         Para.kpar.kp1 =2.^power2; Para.kpar.kp2 = 0;
%         for power1 = 0
%             Para.v_sig = 2.^power1;
%             for j = pa.min:pa.step:pa.max
%                 Para.p1=2.^j;
%                 for tao=pa.taomin:pa.taostep:pa.taomax
%                     Para.p3=tao;
%                     for i = 1:k
%                         test = (indices == i); train = ~test;
%                         Trn.X = X_train_T(train,:); Trn.Y = Y_train_T(train,:);
%                         ValX = X_train_T(test,:); ValY = Y_train_T(test,:);
%                         %---------- V & P ----------
%                         if sum(Vtype=='V')
%                             [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
%                         else
%                             Para.V=eye(size(Trn.X,1),size(Trn.X,1));
%                         end
%                         Para.P=zeros(size(Trn.X,1),size(Trn.X,1));
%                         [P,~] = DPcaculate(Trn.X, Trn.Y,data_ori.X_train_A, PredictY_T(train,:),data_ori.Y_train_A, DV(train,:), KP, pt, Para.p3);
%                         Para.P=P;
%                         %---------- Model ----------
%                         [PredictY , model] = LUSI_VSVM(ValX , Trn , Para);
%                         PredictY.tst(PredictY.tst==-1)=0;
%                         M_Acc(i) = sum(PredictY.tst==ValY)/length(ValY)*100;
%                         CM = ConfusionMatrix(PredictY.tst,ValY);  M_F(i)=CM.FM;
%                     end
%                 end
%                 m =mean(M_Acc); mean_F=mean(M_F);
%                 if  m>t       % mean_Acc>test_ac or mean_F>test_F
%                     t=m       ; 
%                 end 
%             end
%         end
%     end
%     Select(num)=t;
% end
% [value,indice]=sort(Select,"descend");
PT=PT_old;
% >>>>>>>>>>>>>>>>>>>>Setp 2-2<<<<<<<<<<<<<<<<<<<<
Tp="Zero";
for num=1:9
    ptype=Tp;
    for power2 = pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power2; Para.kpar.kp2 = 0;
        for power1 = 0
            Para.v_sig = 2.^power1;
            for j = pa.min:pa.step:pa.max
                Para.p1=2.^j;
                for tao=pa.taomax%pa.taomin:pa.taostep:pa.taomax
                    Para.p3=tao;
                    for i = 1:k
                        test = (indices == i); train = ~test;
                        Trn.X = X_train_T(train,:); Trn.Y = Y_train_T(train,:);
                        ValX = X_train_T(test,:); ValY = Y_train_T(test,:);
                        % ---------- V & P ----------
                        if sum(Vtype=='V')
                            [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
                        else
                            Para.V=eye(size(Trn.X,1),size(Trn.X,1));
                        end
                        Para.P=zeros(size(Trn.X,1),size(Trn.X,1));
                        for u = 1:size(Tp,2)
                            [P,~] = DPcaculate(Trn.X, Trn.Y,data_ori.X_train_A, PredictY_T(train,:),data_ori.Y_train_A, DV(train,:), KP, Tp(u), Para.p3);
                            Para.P=Para.P+P;
                        end
                        Para.P=Para.P./size(Tp,2);
                        % ---------- Model ----------
                        [PredictY , model] = LUSI_VSVM(ValX , Trn , Para);
                        PredictY.tst(PredictY.tst==-1)=0;
                        M_Acc(i) = sum(PredictY.tst==ValY)/length(ValY)*100;
                        CM = ConfusionMatrix(PredictY.tst,ValY);  M_F(i)=CM.FM;
                    end
                end
                mean_Acc =mean(M_Acc); mean_F=mean(M_F);
                if  mean_Acc>test_ac       % mean_Acc>test_ac or mean_F>test_F
                    test_ac=mean_Acc       ; best_v_sig=Para.v_sig  ;
                    best_kp1=Para.kpar.kp1 ; best_p1=Para.p1;best_p3=Para.p3;
                end
            end
        end
    end
    % >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
    Trn.X=X_train_T; Trn.Y=Y_train_T;
    Para.v_sig=best_v_sig;   Para.kpar.kp1=best_kp1; Para.p1=best_p1;Para.p3=best_p3;
    [V,~] = Vmatrix(X_train_T,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
    Para.P=zeros(size(X_train_T,1),size(X_train_T,1));
    for u = 1:size(Tp,2)
        [P,~] = DPcaculate(X_train_T,Y_train_T, data_ori.X_train_A, PredictY_T,data_ori.Y_train_A,DV, KP, Tp(u), Para.p3);
        Para.P=Para.P+P;
    end
    Para.P=Para.P./size(Tp,2);
    [PredictY0 , ~] = LUSI_VSVM(X_train_T , Trn , Para);
    PredictY0.tst(PredictY0.tst==-1)=0;
    pp=PT(num);
    phi = DPcaculate_all(X_train_T,Y_train_T, data_ori.X_train_A, PredictY_T,data_ori.Y_train_A, DV, KP, pp, Para.p3);
    %计算T值
    T(num)=abs(sum(phi'*PredictY0.tst)-sum(phi'*Y_train_T))./(sum(phi'*Y_train_T));
    if T(num)>=median(T(1:end-1))
        Tp=[Tp,PT(num)];
    end
    Para.v_sig=best_v_sig;   Para.kpar.kp1=best_kp1; Para.p1=best_p1;Para.p3=best_p3; best_Tp=Tp;
    fprintf('Complete %.2f %%\n',(num)*100/9)
end
% ---------- V & P----------
if sum(Vtype=='V')
    [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker); Para.V=V;
else
    Para.V=eye(size(Trn.X,1),size(Trn.X,1));  Para.V=eye(size(Trn.X,1),size(Trn.X,1));
end
for u = 1:size(best_Tp,2)
    [P,~] = DPcaculate(Trn.X, Trn.Y,data_ori.X_train_A, PredictY_T,data_ori.Y_train_A,DV, KP, best_Tp(u), Para.p3);
    Para.P=Para.P+P;
end
Para.P=Para.P./size(best_Tp,2);
%                         Para.P=Pcaculate(Trn.X , Trn.Y,ptype);
% ---------- Model ----------
[PredictY , model] = LUSI_VSVM(X_test_T , Trn , Para);
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
fprintf('Best_tao=%.2f||',best_p3)   ;
for t=1:size(best_Tp,2)
    fprintf('Best_Tp=%s||',best_Tp(t));
end
fprintf('\n%s\n', repmat('=', 1, 100));
end

