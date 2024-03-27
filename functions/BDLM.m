%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         act
% Description:  Activation function
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 12, 2019
% Updated:      October 20, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef BDLM  
    methods (Static)
        function bdlm = build(bdlm, sQ, sV, inix, iniSx)
            comp = bdlm.comp;
             for i = 1:length(comp)
                 if comp(i) == 11 % local level
                     A{i} = 1;
                     Q{i} = sQ(i)^2;
                     F{i} = 1;
                     hs{i} = "Local level";
                 elseif comp(i) == 12 % local trend
                     A{i} = [1 1;0 1];
                     Q{i} = sQ(i)^2*[1/3 1/2;1/2 1];
                     F{i} = [1 0];
                     hs{i} = ["Local level", "Local trend"];
                 elseif comp(i) == 13 % local acceleration
                     A{i} = [1 1 0.5;0 1 1; 0 0 1];
                     Q{i} = sQ(i)^2*[1/20 1/8 1/6;1/8 1/3 1/2;1/6 1/2 1];
                     F{i} = [1 0 0];
                     hs{i}= ["Local level", "Local trend", "Local acceleration"];
                 elseif comp(i) == 14 % local jerk
                     A{i} = [1 1 0.5 1/6;0 1 1 0.5; 0 0 1 1; 0 0 0 1];
                     Q{i} = 0*ones(4,4);
                     F{i} = [1 0 0 0];
                     hs{i}= ["Local level", "Local trend", "Local acceleration", "Local jerk"];
                 elseif comp(i) == 7  % LSTM
                     A{i} = 0;
                     Q{i} = sQ(i)^2;
                     F{i} = [1];
                     hs{i} = "LSTM";
                 elseif comp(i) == 113 % local level, trend, exponential smoothing
                     A{i} = [1 1 0 0 0 0;0 1 0 0 0 0; 0 0 1 0 1 0; 0 0 0 1 0 0; zeros(1,6);zeros(1,6)];
                     Q{i} = diag([0;0;0;0;0;0]);
                     F{i} = [1 0 1 0 0 1];
                     hs{i} = ["LL","LT","ESM","alpha","sigma(alpha_{t-1})*v_{t-1}","v"];
                 end
             end
             bdlm.A = blkdiag(A{:});
             bdlm.Q = blkdiag(Q{:});
             bdlm.F = [F{:}];
             bdlm.R = diag(sV.^2);
             bdlm.x  = inix;
             bdlm.Sx = iniSx;
             bdlm.hs = [hs{:}];
        end
        function [yPd, SyPd, xu, Sxu, xp, Sxp] = KF(y, bdlm)
            xp    = zeros(size(bdlm.x,1), numel(y));
            Sxp   = zeros(size(bdlm.x,1)^2, numel(y));
            xu    = xp;
            Sxu   = Sxp;
            yPd    = zeros(1, numel(y));
            SyPd   = zeros(1, numel(y));
            xloop  = bdlm.x;
            Sxloop = bdlm.Sx;
            for i = 1:numel(y)
                [xploop, Sxploop, yploop, Syploop] = BDLM.KFpre(xloop, Sxloop, bdlm.A, bdlm.F, bdlm.Q, bdlm.R);
                xp(:,i)  = xploop;
                Sxp(:,i) = Sxploop(:);
                if ~isnan(y(i))
                    [xuloop, Sxuloop, yuloop, Syuloop] = BDLM.KFup (y(i), xploop, Sxploop, bdlm.F, bdlm.R);
                    xu(:,i)  = xuloop;
                    Sxu(:,i) = Sxuloop(:);
                    xloop     = xuloop;
                    Sxloop    = Sxuloop;
                    yPd(:,i)   = yuloop;
                    SyPd(:,i)  = Syuloop;
                else
                    xu(:,i)  = xploop;
                    Sxu(:,i) = Sxploop(:);
                    xloop  = xploop;
                    Sxloop = Sxploop;
                    yPd(:,i)   = yploop;
                    SyPd(:,i)  = Syploop;
                end
            end
        end
        function [xpre, Sxpre, ypre, Sypre]= KFpre(x, Sx, A, F, Q, R)
            Sx = reshape(Sx, size(x,1),[]);
            xpre  = A*x;
            Sxpre = A*Sx*A'+ Q;
            ypre  = F*xpre;
            Sypre = F*Sxpre*F'+ R;
        end

        function [xpre, Sxpre, ypre, Sypre] = KFPreHybrid(x, Sx, A, F, Q, R, xlstm, Sxlstm)
            Sx = reshape(Sx,size(x,1),[]);
            xpre  = A*x;
            Sxpre = A*Sx*A'+ Q;
            xpre(end)  = xlstm;
            Sxpre(end) = Sxlstm + Q(end);
            ypre  = F*xpre;
            Sypre = F*Sxpre*F' + R;
            Sxpre = Sxpre(:);
        end
        function [xup, Sxup, yup, Syup, deltaMx, deltaVx] = KFup (y, xpre, Sxpre, F, R)
            if any(isnan(y))
                Sxpre = reshape(Sxpre, size(xpre,1),[]);
                idx_nan = find(isnan(y));
                F(idx_nan,:) = [];
                R(idx_nan,:) = [];
                R(:,idx_nan) = [];
                y(idx_nan) = [];
                Sypre  = F*Sxpre*F' + R;
                cov_xy = F*Sxpre;
                %
                deltaMx = cov_xy'/Sypre*(y-F*xpre);
                deltaVx = -cov_xy'/Sypre*cov_xy;
                %
                xup   = xpre + deltaMx;
                Sxup  = Sxpre + deltaVx;
                Sxup  = (Sxup + Sxup')/2;
                %
                yup  = F*xup;
                Syup = F*Sxup*F' + R;
                Sxup = Sxup(:);
            else
                Sxpre = reshape(Sxpre, size(xpre,1),[]);
                Sypre  = F*Sxpre*F' + R;
                cov_xy = Sxpre*F';
                %
                deltaMx = cov_xy/Sypre*(y-F*xpre);
                deltaVx = -cov_xy/Sypre*cov_xy';
                %
                xup   = xpre + deltaMx;
                Sxup  = Sxpre + deltaVx;
                Sxup  = (Sxup + Sxup')/2;
                %
                yup  = F*xup;
                Syup = F*Sxup*F' + R;
                Sxup = Sxup(:);
            end
        end
        function [xsm, Sxsm] = KFSmoother_cov(xpre, Sxpre, xup, Sxup, A, A_hybrid)
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            if isempty(A_hybrid)
                for i = size(xpre,2)-1:-1:1
                    J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
                    %                     J = reshape(Sxup(:,i),[],n)*A'/(reshape(Sxpre(:,i+1),[],n) + 1E-8*eye(size(xpre,1)));
                    xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                    S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                    Sxsm(:,i) = S(:);
                end
            else
                for i = size(xpre,2)-1:-1:1
                    A(end) = A_hybrid(i+1);
                    J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
                    %                     J = reshape(Sxup(:,i),[],n)*A'/(reshape(Sxpre(:,i+1),[],n) + 1E-8*eye(size(xpre,1)));
                    xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                    S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                    Sxsm(:,i) = S(:);
                end
            end
        end
        function [xsm, Sxsm, inix, iniSx] = KFSmoother(xpre, Sxpre, xup, Sxup, A, A_hybrid, inix, iniSx)
            xpre  = [inix, xpre];
            Sxpre = [iniSx(:), Sxpre];
            xup   = [inix, xup];
            Sxup  = [iniSx(:), Sxup];
            A_hybrid = [0; A_hybrid];
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            if isempty(A_hybrid)
                for i = size(xpre,2)-1:-1:1
                    J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
                    xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                    S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                    Sxsm(:,i) = S(:);
                end
            else
                for i = size(xpre,2)-1:-1:1
                    A(end) = A_hybrid(i+1);
                    J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
                    xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                    S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                    Sxsm(:,i) = S(:);
                end
            end
            inix = xsm(:,1);
            iniSx = Sxsm(:,1);
            xsm(:,1)  = [];
            Sxsm(:,1) = [];
        end
       
        % Exponential smoothing with BNI
        function [xpre, Sxpre, ypre, Sypre] = KFPreHybrid_ESM_BNI(comp, x, Sx, A, F, Q, R, xlstm, Sxlstm)
            xlstm  = xlstm(1);
            Sxlstm = Sxlstm(1);
            Sx = reshape(Sx,size(x,1),[]);
            xpre  = A*x;
            Sxpre = A*Sx*A'+ Q;
            xpre(end)  = xlstm;
            Sxpre(end) = Sxlstm;
            ypre  = F*xpre;
            Sypre = F*Sxpre*F' + R;
            Sxpre = Sxpre(:);
        end
        function [xup, Sxup, yup, Syup, deltaMx, deltaVx] = KFup_ESM_BNI (comp, idxV, y, xpre, Sxpre, F, R, mv2b, Sv2b, mv2bt, Sv2bt, cov_V2b_V2bt)
            if any(isnan(y))
                Sxpre = reshape(Sxpre, size(xpre,1),[]);
                idx_nan = find(isnan(y));
                F(idx_nan,:) = [];
                R(idx_nan,:) = [];
                R(:,idx_nan) = [];
                y(idx_nan) = [];
                Sypre  = F*Sxpre*F' + R;
                cov_yx = F*Sxpre;
                %
                deltaMx = cov_yx'/Sypre*(y-F*xpre);
                deltaVx = -cov_yx'/Sypre*cov_yx;
                %
                xup   = xpre + deltaMx;
                Sxup  = Sxpre + deltaVx;
                Sxup  = (Sxup + Sxup')/2;
                %
                yup  = F*xup;
                Syup = F*Sxup*F' + R;
                Sxup = Sxup(:);
            else
                Sxpre = reshape(Sxpre, size(xpre,1),[]);
                Sypre  = F*Sxpre*F' + R;
                cov_yx = F*Sxpre;
                %
                deltaMx = cov_yx'/Sypre*(y-F*xpre);
                deltaVx = -cov_yx'/Sypre*cov_yx;
                %
                xup   = xpre + deltaMx;
                Sxup  = Sxpre + deltaVx;
                Sxup  = (Sxup + Sxup')/2;
                if any(ismember(comp,[113]))
                    % Case 1. using sigma(alpha_{t-1}) as the exponential smoothing
                    % coefficient (same as in the thesis) to have 0< alpha =<1
                    alpha  = xup(4);
                    S_alpha = Sxup(4,4);
                    [sigma_Alpha, var_sigma_Alpha, J_sigma_Alpha] = act.meanVar(alpha, alpha, S_alpha, 2, 1, 1, 0);
                    cov_SigAlpha_x = J_sigma_Alpha*Sxup(4,:);
                    xup(5) = sigma_Alpha*xup(6) + cov_SigAlpha_x(6);
                    Sxup(5,:) = cov_SigAlpha_x*xup(5) + Sxup(5,:)*sigma_Alpha;
                    Sxup(:,5) =  Sxup(5,:);
                    Sxup(5,5) =  var_sigma_Alpha*Sxup(6,6) + cov_SigAlpha_x(6)^2 + 2*cov_SigAlpha_x(6)*sigma_Alpha*xup(6) + var_sigma_Alpha*xup(6)^2 + Sxup(6,6)*sigma_Alpha^2;
                    
                    % Case 2. alpha_{t-1} as the exponential smoothing
                    % coefficient (relax the requirement that 0< alpha =<1). 
                    % Both cases work, but it The Time series forecasting
                    % book indicated that this condition can be relaxed.

%                     xup(5) = xup(4)*xup(6) + Sxup(4,6);
%                     Sxup(5,:) = Sxup(4,:)*xup(6) + Sxup(6,:)*xup(4);
%                     Sxup(:,5) =  Sxup(5,:);
%                     Sxup(5,5) =  Sxup(4,4)* Sxup(6,6) + Sxup(4,6)^2 + 2*Sxup(4,6)*xup(3)*xup(6) + Sxup(4,4)*xup(6)^2 + Sxup(6,6)*xup(4)^2;
                end
                %
                if ~isempty(idxV)
                    mvUd = xup(idxV);
                    SvUd = Sxup(idxV,idxV);
                    [deltaMv2b, deltaSv2b] = BDLM.noiseBackwardUpdateHete (mvUd, SvUd, mv2b, Sv2b, mv2bt, Sv2bt, cov_V2b_V2bt);
                    deltaMx(idxV) = deltaMv2b;
                    deltaVx(idxV,idxV) = deltaSv2b;
                end
                %
                yup  = F*xup;
                Syup = F*Sxup*F' + R;
                Sxup = Sxup(:);
            end
        end
        function [xsm, Sxsm] = KFSmoother_ESM_BNI(comp, xpre, Sxpre, xup, Sxup, A, Czz)
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
                for i = size(xpre,2)-1:-1:1
                    cov_xt1_xt = reshape(Sxup(:,i),[],n)*A';
                    cov_xt1_xt(end) = Czz(i);
                    J = cov_xt1_xt/(reshape(Sxpre(:,i+1),[],n) + 1E-12*eye(size(cov_xt1_xt,1)));
                    xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                    S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                    S  = (S + S')/2;
                    Sxsm(:,i) = S(:);
                end
        end

        function [m_exp,S_exp,cov_exp] = expNormal (m,S)
            m_exp = exp(m + 0.5*S);
            S_exp = exp(2*m+S)*(exp(S)-1);
            cov_exp = S*exp(m+0.5*S);
        end
        function [deltaMv2b, deltaSv2b] = noiseBackwardUpdateHete (mvUd, SvUd, mv2b, Sv2b, mv2bt, Sv2bt, cov_v2b_v2bt)
            gpu = 0;
            mv2Ud = mvUd.^2 + SvUd;
            Sv2Ud = 2*SvUd.^2 + 4*(mvUd.^2).*SvUd;
            [deltaMv2bt, deltaSv2bt] = tagi.noiseBackwardUpdate(mv2bt, 3*Sv2bt + 2*mv2bt.^2, Sv2bt, mv2Ud, Sv2Ud, gpu);
            mv2btUd = mv2bt + deltaMv2bt;
            Sv2btUd = Sv2bt + deltaSv2bt;
            [deltaMv2b, deltaSv2b] = tagi.noiseBackwardUpdate(mv2bt, Sv2bt, cov_v2b_v2bt, mv2btUd, Sv2btUd, gpu);
        end

    end
end