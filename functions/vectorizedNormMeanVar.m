function [mz, Sz] = vectorizedNormMeanVar(ma, Sa, mhat, mw, Sw, mb, Sb, A)
mz  = sqrt(A).*(ma - mhat).*mw + mb;
Sz  = A.*(Sa.*(mw.^2) + Sw.*(ma.^2 - mhat.^2 + Sa)) + Sb;
end