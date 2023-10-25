clear
clc
close all
diary 'Diary.txt'
ttic=tic;
Para. AutoRec = "ON";
%% ==== Ⅰ Running Setting ====
% ----------◆ -1- Model Selecting ◆----------
ModS = [];
ModS = [ModS;"LIB_L1SVC"];
ModS = [ModS;"DLUSI_VSVM2"];
ModS = [ModS;"LSSVM"];
ModS = [ModS;"DASVM"];
ModS = [ModS;"A_LSSVM"];
ModS = [ModS;"CORAL_LSSVM"];
% ----------◆ -2- P Styles ◆----------
P=["A_Y","DV","Kernel","PCA",...
    "Zero","A_Y+DV+Kernel","A_Y+Kernel+PCA"];
% ----------◆ -3- Feature Kernel types ◆----------
Para.kpar.ktype = 'rbf'; % poly or lin or rbf;
% ----------◆ -4- V Kernel types ◆----------
V_Matrix = 'Vmatrix'  ; V_MatrixFun = str2func(V_Matrix) ; Para.vmatrix = V_Matrix;
Para.CDFx = 'uniform' ; Para.v_ker = 'gaussian_ker'      ; CDFx = Para.CDFx;
% ----------◆ -5- Files ◆----------
name= ["diff miu diff sigma"];Para.name=name;
% ----------◆ -6- Repeat ◆----------
Repeat=1;
% ----------◆ -7- K-Fold ◆----------
k=3;
% ----------◆ -8- Para Range ◆----------
pa.min = -8  ;  pa.step =  2 ;  pa.max = 8;
% ----------◆ -9- Base Setting 1 ◆----------
ac_test=[];F_test=[];error_rate_test=[];M_F=[];Prec=[];recall=[];
M_F=[];M_Acc=[];M_Erro=[];M_Errorate=[];sen=[];spe=[] ;GM=[];
%% ==== Ⅱ Running Procedure ====
for pp=1
    ptype=P(pp);
    for ms = 1 : length(ModS)
        Mod = ModS(ms); res.Mod=Mod;
        for chongfu=1:Repeat
            randomindex=randperm(1000,20); seed=randomindex(chongfu); res.seed=seed; rng(seed); res.chongfu=chongfu;
            for l = 1:length(name)
                f = 'data/';  G = name(l) ;  folder = f+G;  files = dir(fullfile(folder, '*.mat'));
                for p = 1:length(files)
                    % ----------◆ -1- Base Setting 2 ◆----------
                    filename = fullfile(folder, files(p).name) ; data_ori = load(filename) ; SVMFun = str2func(Mod);
                    error_test=0;error_rate_test=0;error_lower=0;test_ac=0; test_F=0; test_GM=0 ;
                    [X_train_A,Y_train_A,X_test_A,Y_test_A] = TT(data_ori.X_train_A,data_ori.Y_train_A,0.4);
                    [X_train_T,Y_train_T,X_test_T,Y_test_T] = TT(data_ori.X_test ,data_ori.Y_test ,0.9);
                    fprintf('%s\n', repmat('=', 1, 100)); fprintf('Proc ===>%s\t\n',num2str(chongfu));fprintf('File===>%s\t\n',G);
                    fprintf('Seed===>%s\t\n',num2str(seed)); fprintf('Mod===>%s\t\n',Mod); fprintf('%s\n', repmat('-', 1, 100));
                    
                    % ----------◆ -2- Different Models ◆----------
                    if sum(Mod=='LIB_L1SVC')
                        %---Choose different training data---
                            % X_train=X_train_T;Y_train=Y_train_T;
                            % X_train_T=data_ori.X_train_A; Y_train_T=data_ori.Y_train_A;
                            X_train=[data_ori.X_train_A;X_train_T]; Y_train=[data_ori.Y_train_A;Y_train_T];
                        %------------------------------------    
                        Data.X_train_T=X_train;
                        Data.Y_train_T=Y_train;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = SVM_rbf(Data,k,Para.kpar.ktype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=='DLUSI_VSVM2')
                        pa.Vtype=["I"];% or ["V"]
                        pa.taomin=0.1;
                        pa.taomax=0.9;
                        pa.taostep=0.4;
                        Data.X_train_A=data_ori.X_train_A;
                        Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = DLUSI_VSVM_F(Data,k,Para.kpar.ktype,Para.v_ker,CDFx,ptype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=="LSSVM")
                        %---Choose different training data---
                            % X_train=X_train_T;Y_train=Y_train_T;
                            % X_train=data_ori.X_train_A;Y_train=data_ori.Y_train_A;
                            X_train=[data_ori.X_train_A;X_train_T]; Y_train=[data_ori.Y_train_A;Y_train_T];
                        %------------------------------------
                        Data.X_train_T=X_train;
                        Data.Y_train_T=Y_train;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = LSSVM_F(Data,k,Para.kpar.ktype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=="DASVM")
                        Data.X_train_A=data_ori.X_train_A;
                        Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_test=data_ori.X_test;
                        Data.Y_test=data_ori.Y_test;
                        Res = DASVM_F(Data,k,Para.kpar.ktype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=='A_LSSVM')
                        Data.X_train_A=data_ori.X_train_A;
                        Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        pa.kernel1='gaussian';
                        Res = ALSSVM_F(Data,k,Para.kpar.ktype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=='CORAL_LSSVM')
                    Xs=data_ori.X_train_A;Xt=data_ori.X_test;
                    [Xs_new] = CORAL(Xs,Xt);
                    %LSSVM
                    Data.X_train_T=[Xs_new;X_train_T];
                    Data.Y_train_T=[data_ori.Y_train_A;Y_train_T];
                    Data.X_test_T=X_test_T;
                    Data.Y_test_T=Y_test_T;
                    Res = LSSVM_F(Data,k,Para.kpar.ktype,pa);
                    result=catstruct(res,Res);
                    end
                    % ============================================================
                end
            end
            %% ==== Ⅲ Result Display ====
            ResultsInfo(result,Para)
            MAC(chongfu)=result.ac_test;
            MF(chongfu)=result.F;
            MGM(chongfu)=result.GM;
            MAUC(chongfu)=result.AUC;
        end
        fprintf('AC=%s\t\t',num2str(sprintf('%.2f', mean(MAC))))
        fprintf('AC_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MAC))))
        fprintf('FM=%s\t\t',num2str(sprintf('%.2f', mean(MF))))
        fprintf('F_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MF))))
        fprintf('GM=%s\t\t',num2str(sprintf('%.2f', mean(MGM))))
        fprintf('GM_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MGM))))
        fprintf('AUC=%s\t\t',num2str(sprintf('%.2f', mean(MAUC))))
        fprintf('AUC_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MAUC))))
        fprintf('%s\n', repmat('=', 1, 100))
        clear result
    end
end
toc(ttic)
diary off