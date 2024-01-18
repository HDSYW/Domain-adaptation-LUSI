function  [P]=L3(X,Y)
m=length( Y);
% zheng=X(Y==1,:);
S=sum(X,1)/m;
SS=repmat(S,m, 1);
% SS=S';
G=X-SS;
G=abs(G)';
P=G'*G;
% Y= repmat(Y,1,m); 
Y=Y*Y';
P=Y.*P;
end