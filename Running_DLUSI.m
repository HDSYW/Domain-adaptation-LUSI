clear
clc
close all
Para. AutoRec = "ON";
%% ============================== Ⅰ Running Setting ==============================
% ----------◆ -1- Model Selecting ◆----------
ModS = [];ModS = [ModS;"DLUSI_VSVM"];
% ----------◆ -2- P Styles ◆----------
P=["Zero"];
% ----------◆ -3- ◆----------
datas=[];
datas=[datas;"train"];
% datas=[datas;"test"];
% datas=[datas;"all"];
data=datas(1);
% ----------◆ -4- Feature Kernel types ◆----------
Para.kpar.ktype = 'rbf'; % poly or lin or rbf;
% ----------◆ -5- V Kernel types ◆----------
V_Matrix = 'Vmatrix'  ; V_MatrixFun = str2func(V_Matrix) ; Para.vmatrix = V_Matrix;
Para.CDFx = 'uniform' ; Para.v_ker = 'gaussian_ker'      ; CDFx = Para.CDFx;
% ----------◆ -6- Files ◆----------
name= ["diff miu diff sigma"]; Para.name=name;
% ----------◆ -7- Repeat ◆----------
Repeat=1;
% ----------◆ -8- K-Fold ◆----------
k=3;
% ----------◆ -9- Para Range ◆----------
pa.min = -8  ;  pa.step =  2 ;  pa.max = 8;
%% ============================== Ⅱ Running Procedure ==============================
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
                    [X_train_A,Y_train_A,X_test_A,Y_test_A] = TT(data_ori.X_train_A,data_ori.Y_train_A,0.4);
                    [X_train_T,Y_train_T,X_test_T,Y_test_T] = TT(data_ori.X_test ,data_ori.Y_test ,0.9);
                    fprintf('%s\n', repmat('=', 1, 100)); fprintf('Proc ===>%s\t\n',num2str(chongfu));fprintf('File===>%s\t\n',G);
                    fprintf('Seed===>%s\t\n',num2str(seed)); fprintf('Mod===>%s\t\n',Mod); fprintf('X_C===>%s\t\n',data); fprintf('%s\n', repmat('-', 1, 100));
                    % ============================================================
                    if sum(Mod=='DLUSI_VSVM')
                        pa.Vtype=["I"];
                        pa.taomin=0.1;  pa.taomax=0.9;  pa.taostep=0.4;
                        Data.X_train_A=data_ori.X_train_A;
                        Data.Y_train_A=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = DLUSI_VSVM_F(Data,k,Para.kpar.ktype,Para.v_ker,CDFx,ptype,pa);
                        result=catstruct(res,Res);
                    end
                    % ============================================================
                end
            end
            %% ============================== Ⅲ Result Display ==============================
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