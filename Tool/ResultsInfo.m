function ResultsInfo(result,Para)

    if Para. AutoRec == "ON"
        % |DataSet|Method|rng|AC|SpnR|p1|p2|p3|p4|
%         TIME = datestr(now,'yyyy-mm-dd_HH-MM-SS');
        FilePth = sprintf('./AutoResult/Results_%s.txt',Para.name);
        fid = fopen(FilePth, 'a+'); 
        num=length(fieldnames(result));
        if num==9
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.4f|%.4f|%.4f|%.0f|%.0f|\n',result.seed,result.ac_test,result.F, result.GM, result.AUC,log2(result.lam),log2(result.kp1));
            status=fclose(fid);
        end
        if num==10
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.4f|%.4f|%.4f|%.0f|%.0f|%.0f|\n',result.seed,result.ac_test,result.F, result.GM, result.AUC,log2(result.lam),log2(result.kp1),log2(result.v_sig));
            status=fclose(fid);
        end
        if num==11
            if result.chongfu==1
                fprintf(fid,'%s\n',result.Mod);
            end
            fprintf(fid,'|%.0f|%.2f|%.4f|%.4f|%.4f|%.0f|%.0f|%.0f|\n',result.seed,result.ac_test,result.F, result.GM, result.AUC,log2(result.lam),log2(result.kp1),log2(result.v_sig),log2(result.p4));
            status=fclose(fid);
        end        
        if num==12
            if result.chongfu==1
                fprintf(fid,'%s Ptype=%s\n',result.Mod,result.Ptype);
            end
            fprintf(fid,'|%.0f|%.2f|%.4f|%.4f|%.4f|%.0f|%.0f|%.0f|%.2f|\n',result.seed,result.ac_test,result.F, result.GM, result.AUC,log2(result.kp1),log2(result.lam),log2(result.v_sig),result.tao_1);
            status=fclose(fid);
        end
    end
    
    
end