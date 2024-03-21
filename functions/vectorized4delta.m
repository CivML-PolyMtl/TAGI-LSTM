function [deltaM1, deltaS1, deltaM2, deltaS2] = vectorized4delta(W, C1, C2, deltaM, deltaS)
deltaM1 = W.*C1.*deltaM;
deltaS1 = W.*C1.*deltaS.*W.*C1;
deltaM2 = W.*C2.*deltaM;
deltaS2 = W.*C2.*deltaS.*W.*C2;
end