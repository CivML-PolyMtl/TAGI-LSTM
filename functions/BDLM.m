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
                if comp(i) == 12 % local trend
                    A{i} = [1 1;0 1];
                    Q{i} = sQ(i)^2*[1/3 1/2;1/2 1];
                    F{i} = [1 0];
                    hs{i} = ["Local level", "Local trend"];
                elseif comp(i) == 11 % local level
                    A{i} = 1;
                    Q{i} = sQ(i)^2;
                    F{i} = 1;
                    hs{i} = "Local level";
                elseif comp(i) == 13 % local acceleration
                    A{i} = [1 1 0.5;0 1 1; 0 0 1];
                    Q{i} = sQ(i)^2*[1/20 1/8 1/6;1/8 1/3 1/2;1/6 1/2 1];
                    F{i} = [1 0 0];
                    hs{i}= ["Local level", "Local trend", "Local acceleration"];
                elseif comp(i) == 41 % AR
                    A{i} = bdlm.phiAR;
                    Q{i} = sQ(i)^2;
                    F{i} = 1;
                    hs{i} = "Local level";
                elseif comp(i) == 2  % periodic
                    w = bdlm.w;
                    A{i} = [cos(w) sin(w);-sin(w) cos(w)];
                    Q{i} = sQ(i)^2*[1 0;0 1];
                    F{i} = [1 0];
                    hs{i} = ["periodic1", "periodic2"];
                elseif comp(i) == 7  % LSTM
                    A{i} = 1;
                    Q{i} = sQ(i);
                    F{i} = [1];
                    hs{i} = "LSTM";
                elseif comp(i) == 8 % LSTM + sum
                    A{i} = [1 1;0 1];
                    Q{i} = sQ(i)^2*[0 0;0 0];
                    F{i} = [0 1];
                    hs{i} = ["sum LSTM", "LSTM"];
                end
             end
             bdlm.A = blkdiag(A{:});
             bdlm.Q = blkdiag(Q{:});
             bdlm.F = [F{:}];
             bdlm.R = sV^2;
             bdlm.x  = inix;
             bdlm.Sx = iniSx;
             bdlm.hs = [hs{:}];
        end
        
        function [xpre, Sxpre, ypre, Sypre]= KFpre(x, Sx, A, F, Q, R)
            xpre  = A*x;
            Sxpre = A*Sx*A'+ Q;
            ypre  = F*xpre;
            Sypre = F*Sxpre*F'+ R;
        end
        
        function [xup, Sxup, yup, Syup, deltaMx, deltaVx] = KFup (y, xpre, Sxpre, F, R)
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
            Syup = F*Sxup*F'+R;
        end
        
        function [xsm, Sxsm] = KFSmoother(xpre, Sxpre, xup, Sxup, A)
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            for i = size(xpre,2)-1:-1:1
                J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
                xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                Sxsm(:,i) = S(:);
            end
        end
        
        function [yPd, SyPd, xu, Sxu, xp, Sxp] = KF(y, bdlm)
            xp    = zeros(size(bdlm.x,1), numel(y));
            Sxp   = zeros(size(bdlm.Sx,1)^2, numel(y));
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
        
        function [xpre, Sxpre, ypre, Sypre] = KFpre_Deriv(x, Sx, A, F, Q, R, xLstm, SxLstm)
            xpre  = A*x;
            Sxpre = A*Sx*A'+ Q;
            xpre(end)  = xLstm;
%             Sxpre(end) = Sxpre(end) + SxLstm;
            Sxpre(end) = SxLstm;
            ypre  = F*xpre;
            Sypre = F*Sxpre*F' + R;
        end                
        function [xpre, Sxpre, ypre, Sypre] = KFpre_lstmAR(x, Sx, A, F, Q, R, xLstm, SxLstm, deriv)
            phiAR = A(1,1);
%             xpre  = A*x;
%             Sxpre = A*Sx*A'+ Q;
            xpre(1) = phiAR*x(1);
            Sxpre(1) = phiAR^2*Sx(1);
            xpre(2,1)  = xLstm;
            Sxpre(2,2) = SxLstm;         
            cov_z0xAR = deriv*Sx(1,2);
            cov_z0xAR = phiAR*cov_z0xAR;
            cov_z0xAR = sqrt(SxLstm)/(deriv*sqrt(Sx(2,2)))*cov_z0xAR;
            Sxpre(1,2) = cov_z0xAR;
            Sxpre(2,1) = cov_z0xAR;
            Sxpre = Sxpre + Q;
            ypre  = F*xpre;
            Sypre = F*Sxpre*F' + R;
        end 
        
        function [xsm, Sxsm] = KFSmoother_Deriv(xpre, Sxpre, xup, Sxup, A, deriv)
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end); 
            Sxsm(:,end) = Sxup(:,end);
            for i = size(xpre,2)-1:-1:1
                A(end) = deriv(i+1);
                J = reshape(Sxup(:,i),[],n)*A'/reshape(Sxpre(:,i+1),[],n);
%                 J = reshape(Sxup(:,i),[],n)*A';
%                 J(end) = J(end) + Sxpre(end,i+1);
%                 J = J/reshape(Sxpre(:,i+1),[],n);
                xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                Sxsm(:,i) = S(:);
            end
        end
        function [xsm, Sxsm] = KFSmoother_lstmAR(xpre, Sxpre, xup, Sxup, A)
            n    = size(xpre,1);
            xsm  = zeros(size(xpre));
            Sxsm = zeros(size(Sxpre));
            xsm(:,end) = xup(:,end); 
            Sxsm(:,end) = Sxup(:,end);
            phiAR  = A(1,1);
            for i = size(xpre,2)-1:-1:1
                St1 = Sxpre(4,i+1);
                St  = Sxup(4,i);
                J_ = [phiAR,sqrt(St1/St)].*reshape(Sxup(:,i),[],n);
                J = J_/reshape(Sxpre(:,i+1),[],n);
                xsm(:,i)  = xup(:,i) + J*(xsm(:,i+1)-xpre(:,i+1));
                S = reshape(Sxup(:,i),[],n) + J*(reshape(Sxsm(:,i+1),[],n) - reshape(Sxpre(:,i+1),[],n))*J';
                Sxsm(:,i) = S(:);
            end
        end
        function pl(t, y, yPd, SyPd, xB, SxB, nbtrain, epoch)  
            %
            subplot(2,1,1)
            plot(t, y,'r','lineWidth',1);
            hold on
%             plot(t, xB(1,:),'lineWidth',1);
            xline(t(nbtrain),'b--','lineWidth',1);
            plot(t, yPd,'k','lineWidth',1)
            pt = [t' fliplr(t')];
            py = [yPd' + sqrt(SyPd') fliplr(yPd'-sqrt(SyPd'))];
            patch(pt,py,'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2);
            datetick('x','mm-yy');
            xlim([t(1) t(end)]);
            sgtitle (['Epoch #: ' num2str(epoch)])
            %
            subplot(2,1,2)
            n = size(xB,1);
            hold on
            for i = 1:n
                plot(t, xB(i,:),'lineWidth',1);
                py = [xB(i,:) + sqrt(SxB(i+(i-1)*n,:)) fliplr(xB(i,:)-sqrt(SxB(i+(i-1)*n,:)))];
%                 py = [xB(i,:) + sqrt(SxB(i,:)) fliplr(xB(i,:)-sqrt(SxB(i,:)))];
                patch(pt,py,'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2);
            end
            xline(t(nbtrain),'b--','lineWidth',1);
            datetick('x','mm-yy');
            xlim([t(1) t(end)]);
        end
        
        function plotHs(t, y, yPd, SyPd, x, Sx, nbtrain, hs) 
%             idxfig =  findobj('type','figure');
%             idxfig = length(idxfig);
            yPd  = reshape(yPd,1,[]);
            SyPd = reshape(SyPd,1,[]);
            idxfig = 0;
            n = size(x,1);
            for i = 1:n
                figure(i+idxfig)
                hold on
                plot(t, x(i,:), 'k','lineWidth',1)
                pt = [t' fliplr(t')];
                py = [x(i,:) + sqrt(Sx(i+(i-1)*n,:)) fliplr(x(i,:)-sqrt(Sx(i+(i-1)*n,:)))];
                patch(pt,py,'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2);
                datetick('x','mm-yy');
                xlim([t(1) t(end)]);
                xline(t(nbtrain),'b--','lineWidth',1);
                title(char(hs(i)))
            end
            figure(idxfig+n+1)
            hold on
            plot(t, y, 'r','lineWidth',1)
            plot(t, yPd, 'k','lineWidth',1)
            pt = [t' fliplr(t')];
            py = [yPd + sqrt(SyPd) fliplr(yPd-sqrt(SyPd))];
            patch(pt,py,'green','EdgeColor','none','FaceColor','green','FaceAlpha',0.2);
            datetick('x','mm-yy');
            xlim([t(1) t(end)]);
            xline(t(nbtrain),'b--','lineWidth',1);
            title('BDLM prediction - obs')
        end
        
        function bdlm = updateBdlmInitHS (bdlm, initx, initSx)
            initSx = reshape(initSx,[],numel(initx));
            bdlm.x  = initx;
%             bdlm.Sx = initSx;
        end
        
        function xsm = ma(y, w)
%             n = numel(y) - w;
%             xsm = zero(n,1);
            for i = w+1:numel(y) 
               xsm(i) =  sum(y(i-w:i),'all')/(w+1);
            end
        end
    end
end