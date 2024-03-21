function [mz, Sz] = vectorizedMeanVar_cov(ma, mp, Sa, Sp, Cap)
Sz = Sp.*ma.*ma + Sa.*Sp + Sa.*mp.*mp + Cap.^2 + 2.*Cap.*ma.*mp;
mz = ma.*mp + Cap;
end