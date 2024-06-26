%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         network
% Description:  Build networks relating to each task
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef network
    methods (Static)      
        % LSTM
        function [zl, Szl, lstm, zlpri, Szlpri,rnnSmooth] = LSTM(net, theta, normStat, states, maxIdx, Mem, x, y, Sq)
            sql       = net.sql;
            batchSize = net.batchSize;

            % Sequence length
            if isempty(Sq)
                ySq   = y(1:sql,:);
                SySq  = 0.*ones(sql,batchSize,'like',x);
            else
                ySq  = Sq{1};
                SySq = Sq{2};
            end
            numObs   = size(y,1);
            nblayer = length(net.layer);
            % Save for smoothing
            Chh    = cell(nblayer,numObs);  % cov(h_t,h_{t-1}): covariance between hiddens states of t and t-1
            Ccc    = cell(nblayer,numObs);  % cov(c_t,c_{t-1}): covariance between cell states of t and t-1
            Cxh    = cell(nblayer,numObs);
            Czz    = nan(numObs,1);         % cov(z^{O}_{t-1},z^{O}_t)
            cPrior  = cell(nblayer,numObs); % Prior of cell states
            ScPrior = cell(nblayer,numObs);
            hPrior  = cell(nblayer,numObs); % Prior of hidden states
            ShPrior = cell(nblayer,numObs);
            cPos  = cell(nblayer,numObs);   % Posterior of cell states
            ScPos = cell(nblayer,numObs);
            hPos  = cell(nblayer,numObs);   % Posterior of hidden states
            ShPos = cell(nblayer,numObs);

            % Prepare input for LSTM
            % xSqloop = [covariates ; sequence length (history of target time series)] 
            if ~isempty(x)
                xSq  = x(1,:,:);
                SxSq = zeros(size(xSq)); 
                xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                xSqloop(isnan(xSqloop)) = 0;
                SxSqloop  = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
            else
                xSqloop  = ySq;
                SxSqloop = SySq;
            end
            
            mem0 = Mem;
            %
            numDataPerBatch = net.repBatchSize*net.batchSize;
            zl      = zeros(numObs, net.ny*net.batchSize, 'like',x);
            Szl     = zeros(numObs, net.ny*net.batchSize, 'like',x);
            zlpri   = zl;
            Szlpri  = Szl;

            for i = 1:numObs
                idxBatch = i;
                
                % Prepare output
                ySqloop = y(idxBatch, :)';
                
                % Training and testing
                if strcmp(net.RNNtype, 'LSTM_lookback') w = 1; else w = sql; end
                for k = 1:1:w
                    % Loading memory
                    if strcmp(net.RNNtype, 'LSTM_lookback')
                        xloop   = xSqloop(:);
                        Sxloop  = SxSqloop(:);
                        yloop   = ySqloop(:);
                        mem     = Mem(1:4,1);
                    end
                    states  = tagi.lstmInitializeInputs(states, xloop, Sxloop, [], [], [], [], [], [], [], net.xsc, mem);
                    % Training
                    if net.trainMode == 1
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            % Estimate priors
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            [~, ~, ~, cPrior(:,i) , ScPrior(:,i), ~, ~] = tagi.lstmExtractStates(states); % extract the prior for cell states
                            hPrior(:,i)  = ma; % extract the prior for hidden states
                            ShPrior(:,i) = Sa;
                            zlpri(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])'; % extract the prior for z^{O}
                            Szlpri(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';
                            if any(~isnan(yloop)) % when having observations
                                [deltaM, deltaS, deltaMx, deltaSx, ~, ~, deltaHm, deltaHs, Chh(:,i), Ccc(:,i), Cxh(:,i)] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                if net.batchSize == 1
                                    Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                                end
                                dtheta = tagi.parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                                theta  = tagi.globalParameterUpdate(theta, dtheta, net.gpu);
                                % Estimate posteriors:
                                % deltaHm and deltaHs: are the delta for means and variances to update the activation
                                % units; they are needed to estimate the
                                % posteriors for states
                                states = tagi.lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, states{16}{5});
                                [~, ~, ma, Sa]   = tagi.extractStates(states);
                                [~, ~, ~, cPos(:,i) , ScPos(:,i), ~, ~] = tagi.lstmExtractStates(states); % extract the posterior for cell states
                                hPos(:,i)  = ma; % extract the posterior for hidden states
                                ShPos(:,i) = Sa;
                            else  % when missing data
                                % posteriors = priors
                                [~, ~, ~, ~, ~, ~, ~, ~, Chh(:,i), Ccc(:,i), Cxh(:,i)] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                if net.batchSize == 1
                                    Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                                end
                                cPos(:,i)  = cPrior(:,i); % cell states: posterior = prior
                                ScPos(:,i) = ScPrior(:,i);
                                hPos(:,i)  = hPrior(:,i); % hidden states: posterior = prior
                                ShPos(:,i) = ShPrior(:,i);
                            end

                            %  Posterior for z^{O}
                            zl(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])';
                            Szl(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';

                            % Prepare input for the next time step:
                            if strcmp(net.teacherForcing,'zO')
                                ySq      = cat(1, ySq(2:end,:), zl(idxBatch, :));
                                SySq     = cat(1,SySq(2:end,:), Szl(idxBatch, :));
                            elseif strcmp(net.teacherForcing,'obs')
                                ySq      = cat(1, ySq(2:end,:), y(idxBatch, :));
                            end
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                            %
                        end


                    % Testing: not used for smoothing
                    else
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            if any(~isnan(yloop))
                                [deltaM, deltaS, ~, ~, ~, ~, deltaHm, deltaHs] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                states = tagi.lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, states{16}{5});
                                [~, ~, ma, Sa]   = tagi.extractStates(states);
                            end
                            zl(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])';
                            Szl(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';

                            % Prepare input for the next time step:
                            ySq      = cat(1, ySq(2:end,:), zl(idxBatch, :));
                            SySq     = cat(1,SySq(2:end,:), Szl(idxBatch, :));
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                            SxSqloop(isnan(SxSqloop)) = 0;
                            %
                        end
                    end
                    % Update lstm memory after each timestamp
                    Mem = tagi.updateRnnMemory(net.RNNtype, states);
                end
                if i == 1 % save the input at t=1 for smoothing the sequence
                    x0  = xloop;
                    Sx0 = Sxloop;
                end
            end
            % update lstm memory after each epoch
            lstm.theta = theta;
            lstm.Mem   = Mem;
            
            % save for smoother
            if net.trainMode == 1
                % for cell states
                rnnSmooth.cPrior  = [mem0{3},cPrior];
                rnnSmooth.ScPrior = [mem0{4},ScPrior];
                rnnSmooth.cPos    = [mem0{3},cPos];
                rnnSmooth.ScPos   = [mem0{4},ScPos];
                % for hidden states
                rnnSmooth.hPrior  = [mem0{1},hPrior];
                rnnSmooth.ShPrior = [mem0{2},ShPrior];
                rnnSmooth.hPos    = [mem0{1},hPos];
                rnnSmooth.ShPos   = [mem0{2},ShPos];
                % for z^{O}
                rnnSmooth.zOPos  = zl;
                rnnSmooth.SzOPos = Szl;
                rnnSmooth.zOPrior   = zlpri;
                rnnSmooth.SzOPrior  = Szlpri;
                % for initial LSTM's memory
                rnnSmooth.initMem  = mem0;
                % for covariances cov(h_t,h_{t-1}),  cov(c_t,c_{t-1})
                % cov(x_t,h^{1st LSTM layer}_{t})
                rnnSmooth.Chh = [Chh(:,1), Chh];
                rnnSmooth.Ccc = [Ccc(:,1), Ccc];
                rnnSmooth.Cxh = Cxh;
                rnnSmooth.Czz = Czz;
                % the input x at t=0
                rnnSmooth.x0 = x0;
                rnnSmooth.Sx0 = Sx0;
            else
                rnnSmooth = [];
            end
        end
        % Couple LSTM and SSM
        function [yPd, SyPd, zl, Szl, xBp, SxBp, xBu, SxBu, lstm, Czz] = hydrid_AGVI(net, theta, normStat, states, maxIdx, Mem, x, y, Sq, bdlm)
            % LSTM innitialization
            sql       = net.sql;
            batchSize = net.batchSize;
            if isempty(Sq)
                ySq   = y(1:sql,:);
                SySq  = zeros(size(ySq),'like',x);
                y(1:sql,:) = [];
                if ~isempty(x)
                    x(1:sql,:,:) = [];
                end
            else
                ySq  = Sq{1};
                SySq = Sq{2};
            end
            numObs   = size(y,1);
            nblayer = length(net.layer);
            Chh = cell(nblayer,numObs); % cov(h_t,h_{t-1}): covariance between hiddens states of t and t-1 for smoother
            if ~isempty(x)
                xsql     = net.xsql;
                nbCov    = net.nbCov;
                xSq  = x(1,:,:);
                SxSq = zeros(size(xSq));
                xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                xSqloop(isnan(xSqloop)) = 0;
                SxSqloop  = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
            else
                xSqloop  = ySq;
                SxSqloop = SySq;
            end
            % BDLM innitialization
            xBp  = zeros(size(bdlm.x,1),numObs);
            SxBp = zeros(size(bdlm.x,1)^2,numObs);
            xBu  = zeros(size(bdlm.x,1),numObs);
            SxBu = zeros(size(bdlm.x,1)^2,numObs);
            xBloop  = bdlm.x;
            SxBloop = bdlm.Sx;
            A   = bdlm.A;
            F   = bdlm.F;
            Q   = bdlm.Q;
            R   = bdlm.R;
            yPd   = zeros(numObs, 1, net.dtype);
            SyPd  = zeros(numObs, 1, net.dtype);
            Czz = zeros(numObs,1); % coefficient, to be used in A matrix, at pos. of lstm
            mem0 = Mem;
            if any(ismember(bdlm.comp,[113]))
                idxV = 6;
            end
            %
            numDataPerBatch = net.repBatchSize*net.batchSize;
            zl      = zeros(numObs, net.batchSize, 'like',x);
            Szl     = zeros(numObs, net.batchSize, 'like',x);

            for i = 1:numDataPerBatch:numObs
                idxBatch = i;
                % Prepare output
                ySqloop = y(idxBatch, :)';

                % Training and testing
                if strcmp(net.RNNtype, 'LSTM_lookback') w = 1; else w = sql; end
                for k = 1:1:w
                    % loading memory
                    xloop   = xSqloop(:);
                    Sxloop  = SxSqloop(:);
                    yloop   = ySqloop(:);
                    mem     = Mem(1:4,1);
                    states  = tagi.lstmInitializeInputs(states, xloop, Sxloop, [], [], [], [], [], [], [], net.xsc, mem);
                    % Training
                    if net.trainMode == 1
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            xLp  = reshape(ma{end,1}, [net.ny, numDataPerBatch]);  % x_LSTM prior
                            SxLp = reshape(Sa{end,1}, [net.ny, numDataPerBatch]);  % Sx_LSTM prior  
                            % TAGI-LSTM: covariance between h_{t-1} and h_{t}
                            [Chh(:,i)] = tagi.cov4smoother(net, theta, states);
%                           % TAGI-LSTM: covariance between z^{O}_{t-1} and z^{O}_{t}
                            if numel(xLp)>1
                                Czz(i) = tagi.covZZlstm_2outputs(net, theta, ma, mem, Chh{end-1,i});
                            else
                                Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                            end
                            if any(~isnan(yloop))
                                % Transition step
                                if any(ismember(bdlm.comp,[113]))
                                    mv2b   = xLp(2);
                                    Sv2b   = SxLp(2);
                                    [mv2bt, Sv2bt, cov_v2b_v2bt] = BDLM.expNormal(mv2b,Sv2b);
                                    Q(idxV,idxV) = mv2bt;
                                    [xBp(:,i) , SxBp(:,i)] = BDLM.KFPreHybrid_ESM_BNI(bdlm.comp, xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                    [xBu(:,i), SxBu(:,i), yPd(i,:), SyPd(i,:), deltaMx_, deltaVx_]= BDLM.KFup_ESM_BNI (bdlm.comp, idxV, yloop, xBp(:,i), SxBp(:,i), F, R, mv2b, Sv2b, mv2bt, Sv2bt, cov_v2b_v2bt);
                                    deltaMxL = deltaMx_([length(deltaMx_);idxV]);
                                    deltaSxL = [deltaVx_(length(deltaMx_),length(deltaMx_));deltaVx_(idxV,idxV)];
                                else
                                    [xBp(:,i) , SxBp(:,i)] = BDLM.KFPreHybrid(xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                    [xBu(:,i), SxBu(:,i), yPd(i,:), SyPd(i,:), deltaMx_, deltaVx_]= BDLM.KFup (yloop, xBp(:,i), SxBp(:,i), F, R);
                                    deltaMxL = deltaMx_(end);
                                    deltaSxL = deltaVx_(end);
                                end
                                % Backward-LSTM; update LSTM network
                                [deltaM, deltaS, deltaMx, deltaSx, ~, ~, deltaHm, deltaHs] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, deltaMxL, deltaSxL, [], maxIdx);
                                dtheta = tagi.parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                                theta  = tagi.globalParameterUpdate(theta, dtheta, net.gpu);
                                Cch = states{16}{5}; 
                                states = tagi.lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, Cch);
                            else
                                % Transition step
                                if any(ismember(bdlm.comp,[113]))
                                    mv2b   = xLp(2);
                                    Sv2b   = SxLp(2);
                                    [mv2bt, Sv2bt, cov_v2b_v2bt] = BDLM.expNormal(mv2b,Sv2b);
                                    Q(idxV,idxV) = mv2bt;
                                    [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid_ESM_BNI(bdlm.comp, xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                else
                                    [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid(xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                end
                                xBu(:,i)  = xBp(:,i);
                                SxBu(:,i) = SxBp(:,i);
                            end
                            xBloop  = xBu(:,i);
                            SxBloop = SxBu(:,i);
                            zl(i, :)  = xBloop(end);
                            Szl(i, :) = SxBloop(end);
                            ySq  = cat(1, ySq(2:end,:), zl(i, :));
                            SySq = cat(1,SySq(2:end,:), Szl(i, :));
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                        end

                    % Validation
                    elseif net.trainMode == 2 
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            xLp  = reshape(ma{end,1}, [net.ny, numDataPerBatch]);  % x_LSTM prior
                            SxLp = reshape(Sa{end,1}, [net.ny, numDataPerBatch]);  % Sx_LSTM prior
                            % TAGI-LSTM: covariance between h_{t-1} and h_{t}
                            [Chh(:,i)] = tagi.cov4smoother(net, theta, states);
                            % TAGI-LSTM: covariance between z^{O}_{t-1} and z^{O}_{t}
                            if numel(xLp)>1
                                Czz(i) = tagi.covZZlstm_2outputs(net, theta, ma, mem, Chh{end-1,i});
                            else
                                Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                            end
                            if any(~isnan(yloop))
                                % Transition step
                                if any(ismember(bdlm.comp,[113]))
                                    mv2b   = xLp(2);
                                    Sv2b   = SxLp(2);
                                    [mv2bt, Sv2bt, cov_v2b_v2bt] = BDLM.expNormal(mv2b,Sv2b);
                                    Q(idxV,idxV) = mv2bt;
                                    [xBp(:,i) , SxBp(:,i),yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid_ESM_BNI(bdlm.comp, xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                    [xBu(:,i), SxBu(:,i)]= BDLM.KFup_ESM_BNI (bdlm.comp, idxV, yloop, xBp(:,i), SxBp(:,i), F, R, mv2b, Sv2b, mv2bt, Sv2bt, cov_v2b_v2bt);
                                else
                                    [xBp(:,i) , SxBp(:,i)] = BDLM.KFPreHybrid(xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                    [xBu(:,i), SxBu(:,i), yPd(i,:), SyPd(i,:)]= BDLM.KFup (yloop, xBp(:,i), SxBp(:,i), F, R);
                                end
                            else
                                % Transition step
                                if any(ismember(bdlm.comp,[113]))
                                    mv2b   = xLp(2);
                                    Sv2b   = SxLp(2);
                                    [mv2bt, Sv2bt, cov_v2b_v2bt] = BDLM.expNormal(mv2b,Sv2b);
                                    Q(idxV,idxV) = mv2bt;
                                    [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid_ESM_BNI(bdlm.comp, xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                else
                                    [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid(xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                                end
                                xBu(:,i)  = xBp(:,i);
                                SxBu(:,i) = SxBp(:,i);
                            end
                            xBloop  = xBu(:,i);
                            SxBloop = SxBu(:,i);
                            zl(i, :)  = xBloop(end);
                            Szl(i, :) = SxBloop(end);
                            ySq  = cat(1, ySq(2:end,:), zl(i, :));
                            SySq = cat(1,SySq(2:end,:), Szl(i, :));
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                        end

                    % Testing
                    elseif net.trainMode == 0
                        if k == w
                            [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            xLp  = reshape(ma{end,1}, [net.ny, numDataPerBatch])';  % x_LSTM prior
                            SxLp = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';  % Sx_LSTM prior
                            % TAGI-LSTM: covariance between h_{t-1} and h_{t}
                            [Chh(:,i)] = tagi.cov4smoother(net, theta, states);
                            % TAGI-LSTM: covariance between z^{O}_{t-1} and z^{O}_{t}
                            if numel(xLp)>1
                                Czz(i) = tagi.covZZlstm_2outputs(net, theta, ma, mem, Chh{end-1,i});
                            else
                                Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                            end
                            % Transition step
                            if any(ismember(bdlm.comp,[112 113]))
                                mv2b   = xLp(2);
                                Sv2b   = SxLp(2);
                                [mv2bt, Sv2bt, cov_v2b_v2bt] = BDLM.expNormal(mv2b,Sv2b);
                                Q(idxV,idxV) = mv2bt;
                                [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid_ESM_BNI(bdlm.comp, xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                            else
                                [xBp(:,i), SxBp(:,i), yPd(i,:), SyPd(i,:)] = BDLM.KFPreHybrid(xBloop, SxBloop, A, F, Q, R, xLp, SxLp);
                            end
                            xBu(:,i)  = xBp(:,i);
                            SxBu(:,i) = SxBp(:,i);
                            xBloop  = xBu(:,i);
                            SxBloop = SxBu(:,i);
                            zl(i, :)  = xBloop(end);
                            Szl(i, :) = SxBloop(end);
                            ySq  = cat(1, ySq(2:end,:), zl(i, :));
                            SySq = cat(1,SySq(2:end,:), Szl(i, :));
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                            SxSqloop(isnan(SxSqloop)) = 0;
                        end
                    end
                    % update lstm memory after each timestamp
                    Mem = tagi.updateRnnMemory(net.RNNtype, states);
                end

            end
            % update lstm memory after each epoch
            lstm.theta = theta;
            lstm.Mem   = Mem;
        end
        % Initialization
        function [net, states, maxIdx, netInfo] = initialization(net)
            % Build indices
            net = indices.initialization(net);
            net = indices.layerEncoder(net);
            net = indices.parameters(net);
            net = indices.covariance(net);
            netInfo = indices.savedInfo(net);
            % States
            states = tagi.initializeStates(net.nodes, net.batchSize, net.repBatchSize, net.xsc, net.dtype, net.gpu);
            if strcmp (net.RNNtype, 'LSTM_stateful') || strcmp (net.RNNtype,'LSTM_stateless') || strcmp (net.RNNtype, 'LSTM_lookback')
                states = tagi.lstmInitializeStates(states);
            elseif strcmp (net.RNNtype, 'GRU_stateful') || strcmp (net.RNNtype,'GRU_stateless') || strcmp (net.RNNtype, 'GRU_lookback')
                states = tagi.gruInitializeStates(states);
            end
            maxIdx = tagi.initializeMaxPoolingIndices(net.nodes, net.layer, net.layerEncoder, net.batchSize, net.repBatchSize, net.dtype, net.gpu);   
        end
         
    end
end