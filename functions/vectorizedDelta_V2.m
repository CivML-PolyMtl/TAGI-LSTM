function [deltaM, deltaS] = vectorizedDelta_V2(A, B, deltaM, deltaS)
deltaM = A.*B.*deltaM;
deltaS = (A.*B.*deltaS.*A.*B);
end