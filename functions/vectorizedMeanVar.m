function [mz, Sz] = vectorizedMeanVar(ma, mp, Sa, Sp)
Sz = Sp.*ma.*ma + Sa.*Sp + Sa.*mp.*mp;
mz = ma.*mp;
end