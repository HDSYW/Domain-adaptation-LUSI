function [PredY,model] = LUSI_VSVM(TestX,Data,Para)
%---------- Initiation ----------
gam = Para.p1;%kp
tao =Para.p3;
tao1 = 1-Para.p3;
X = Data.X;
Y = Data.Y;
KerX = KerF(X,Para.kpar,X);
KerTstX = KerF(TestX,Para.kpar,X)';
V = Para.V;
P = Para.P;
%---------- Solve A & b ----------
Ab = inv((tao1*V+tao*P)*KerX+gam*eye(length(Y)))*(tao1*V+tao*P)*Y;
Ac = inv((tao1*V+tao*P)*KerX+gam*eye(length(Y)))*(tao1*V+tao*P)*(ones(length(Y),1));
c = (ones(1,length(Y))*(tao1*V+tao*P)*(Y-KerX*Ab))/(ones(1,length(Y))*(tao1*V+tao*P)*ones(length(Y),1)-ones(1,length(Y))*(tao1*V+tao*P)*KerX*Ac);
A = Ab-c*Ac;
%---------- Output----------
FTrn = (A'*KerX)'+c;%% Y.Trn output
FTst = (A'*KerTstX)'+c;%%Y.test output
% FTst = (A'*KerTstX)'+c+Para.modeldv;%%Y.test output
ftst = FTst-0.5;
ftrn = FTrn-0.5;
f = (A'*KerTstX)'+c;
PredY.tst = sign(ftst);
PredY.trn = sign(ftrn);
model.alpha = A;
model.w = A'*X;
model.b = c;
model.ftrn = ftrn;
model.P = P;
model.f = f;
model.trnX = Data.X;
model.prob=f;
% if Para.drw == 1
%     drw.ds = f;
%     drw.ss1 = drw.ds - 1;
%     drw.ss2 = drw.ds + 1;
%     model.drw = drw;
%     model.twin = 0;
% end
end
