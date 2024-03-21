function [deltaM, deltaS] = vectorizedDelta(C, deltaM, deltaS)
deltaM = C.*deltaM;
deltaS = C.*deltaS.*C;
end