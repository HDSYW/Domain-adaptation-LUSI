function CM = ConfusionMatrix( PredictY, TrueY )
% % Calculate the Confusion Matrix and other related evaluation indicators. 
% %     ┌───┬───────────┐
% %     │　 │  PREDICT  │
% %     ├───┼───┬───┬───┤
% %     │   T   │ │ ++  │  --   │
% %     │   R   ├───┼───┼───┤
% %     │   U   │ ++  │  TP  │ FN  │
% %     │   E   ├───┼───┼───┤
% %     │       │   --  │  FP  │  TN │
% %     └───┴───┴───┴───┘

%% Intermediate Indicators  
    PredictY(PredictY==0)=-1;
    TrueY(TrueY==0)=-1;
    PY = PredictY;      TY = TrueY;   

    TP = nnz( PY(TY==1)==1 );     % # TURE Positive Prediction
    TN = nnz( PY(TY==-1)==-1 ); % # TURE Negative Prediction
    FP = nnz( PY(TY==-1)==1 );   % # FALSE Positive Prediction
    FN = nnz( PY(TY==1)==-1 );  % # FALSE Negative Prediction

    TPTN = TP + TN;  % # TURE Prediction
    FNFP = FN + FP;  % # FALSE Prediction
%         TPFP = TP + FP;   % # Positive Prediction
%         FNTN = FN + TN; % # Negative Prediction
%         TPFN = TP + FN;  % # Positive Label
%         FPTN = FP + TN;  % # Negative Label

    m = length(PY);       
    mp = TP + FN; % nnz(PY==1)  
    mn = FP + TN; % nnz(PY==-1)   
    [ CM.TP , CM.TN , CM.FP , CM.FN ] = deal( TP , TN , FP , FN );
    [ CM.TPTN , CM.FNFP ] = deal( TPTN , FNFP );     
    
%% Final Indicators

    Ac = TPTN / m * 100;
    Er = FNFP / m * 100;
    Sen = TP / mp * 100; 
    Spe = TN / mn * 100; 
    Pre=TP/(TP+FP)* 100;
    GM = sqrt( Sen * Spe ); 
%     FM = (2*Sen*Spe)./(Sen+Spe);
    FM = (2*Sen*Pre)./(Sen+Pre);
    [CM.Ac, CM.Er , CM.Sen , CM.Spe , CM.GM,CM.FM] = deal( Ac , Er , Sen , Spe , GM ,FM);

end


