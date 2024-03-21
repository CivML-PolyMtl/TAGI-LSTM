function [m1, S1, m2, S2] = fourPlus(m1, S1, m2, S2, deltaM1, deltaS1, deltaM2, deltaS2)
m1 = m1 + deltaM1;
S1 = S1 + deltaS1;
m2 = m2 + deltaM2;
S2 = S2 + deltaS2;
end