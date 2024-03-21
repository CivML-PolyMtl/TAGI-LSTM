function [deltaMw, deltaSw, deltaMb, deltaSb] = vectorizedDelta4normParam(Cwz, Cbz, deltaM, deltaS)
deltaMw = Cwz.*deltaM;
deltaSw = Cwz.*deltaS.*Cwz;
deltaMb = Cbz.*deltaM;
deltaSb = Cbz.*deltaS.*Cbz;
end