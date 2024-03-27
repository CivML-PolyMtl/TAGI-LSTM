%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         tagi
% Description:  Tractable Approximate Gaussian Inference (TAGI) 
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 03, 2019
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef tagi
    methods(Static) 
        % Feedforward
        function [states, normStat, maxIdx] = feedForwardPass(net, theta, normStat, states, maxIdx)
            % Initialization
%             net.gpu=1;
            numLayers  = length(net.nodes);
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            if strcmp(net.RNNtype,'LSTM_lookback') || strcmp(net.RNNtype,'LSTM_stateful') || strcmp(net.RNNtype,'LSTM_stateless')
                [mga, Sga, Jga, mc, Sc, Jca, mem] = tagi.lstmExtractStates(states);
                Cch = cell(numLayers, 1);
                Cch{end} = zeros(size(mz{end}),'like',mz{end});
            elseif strcmp(net.RNNtype,'GRU_lookback') || strcmp(net.RNNtype,'GRU_stateful') || strcmp(net.RNNtype,'GRU_stateless')
                Jh  = cell(numLayers, 1);
                [mga, Sga, Jga, mem] = tagi.gruExtractStates(states);
            end
            [mra, Sra] = tagi.extractNormStat(normStat);
            actFunIdx  = net.actFunIdx; 
            layer      = net.layer;
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;   
            mhat       = cell(numLayers, 1);
            Shat       = cell(numLayers, 1);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            % Hidden Layers
            for j = 2:numLayers
                idxw = (numParamsPerlayer_2(1, j-1)+1):numParamsPerlayer_2(1, j);
                idxb = (numParamsPerlayer_2(2, j-1)+1):numParamsPerlayer_2(2, j);     
                % Max pooling
                if layer(j) == net.layerEncoder.mp 
                    maPool = normrnd(ma{j-1}, sqrt(abs(Sa{j-1})));
                    if net.padding(j-1) ~= 0
                        maPool = vertcat(maPool, -Inf*ones(1, size(maPool, 2), 'like', maPool));
                    end
                    maPool(Sa{j-1}<=0) = -Inf;
                    [mz{j}, Sz{j}, maxIdx{j}] = tagi.mpMeanVar(mz{j}, Sz{j}, maPool, ma{j-1}, Sa{j-1}, net.idxPooling{j-1}, maxIdx{j}, rB, net.gpu);
                    
                % Average pooling     
                elseif layer(j) == net.layerEncoder.ap 
                    [mz{j}, Sz{j}] = tagi.apMeanVar(mz{j}, Sz{j}, ma{j-1}, Sa{j-1}, net.idxPooling{j-1}, net.padding(j-1), rB);
                    
                % Normalization     
                elseif layer(j) == net.layerEncoder.ln || layer(j) == net.layerEncoder.bn 
                    if net.trainMode == 1
                        [mhat{j-1}, Shat{j-1}] = tagi.pMeanVar(ma{j-1}, Sa{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                        % Running average for mean and variance
                        mra{j-1} = net.normMomentum*mra{j-1} + (1-net.normMomentum)*mhat{j-1};
                        Sra{j-1} = net.normMomentum*Sra{j-1} + (1-net.normMomentum)*Shat{j-1};
                    end    
                    mhatD = tagi.distributeNormMeanVar(mra{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                    ShatD = tagi.distributeNormMeanVar(Sra{j-1}, nodes(j-1), imgW(j-1), imgH(j-1), filter(j-1), B, rB, layer(j-1), layer(j), net.layerEncoder);
                    if layer(j-1) == net.layerEncoder.fc
                        [mz{j}, Sz{j}] = tagi.fcNormMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, mhatD, ShatD, epsilon, B, rB, net.gpu);
                    elseif layer(j-1) == net.layerEncoder.conv||layer(j-1) == net.layerEncoder.tconv                       
                        [mz{j}, Sz{j}] = tagi.convNormMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, mhatD, ShatD, epsilon, imgH(j-1), imgH(j-1), filter(j-1), B, rB, net.gpu);
                    end  
                    
                % Convolutional
                elseif layer(j) == net.layerEncoder.conv 
                    if B==1&&rB==1
                        [mz{j}, Sz{j}] = tagi.convMeanVarB1(mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                            kernelSize(j-1), filter(j-1), imgW(j), imgH(j), filter(j), net.padding(j-1), net.gpu);
                    else
                        [mz{j}, Sz{j}] = tagi.convMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                            kernelSize(j-1), filter(j-1), imgW(j), imgH(j), filter(j), B, rB, net.padding(j-1), net.gpu);
                    end                                        
                % Transposed convolutional    
                elseif layer(j) == net.layerEncoder.tconv  
                    [mz{j}, Sz{j}] = tagi.tconvMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, net.idxFmwa(j-1, :),...
                        imgW(j), imgH(j), filter(j), B, rB, net.gpu); 
                    
                % Full-connected
                elseif layer(j) == net.layerEncoder.fc
                    if B==1&&rB==1
                        [mz{j}, Sz{j}] = tagi.fcMeanVarB1(mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, nodes(j-1), nodes(j), net.gpu);
                    else
                        [mz{j}, Sz{j}] = tagi.fcMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ma{j-1}, Sa{j-1}, nodes(j-1), nodes(j), B, rB, net.gpu);
                    end
                
                % LSTM
                elseif layer(j) == net.layerEncoder.lstm
                    mem    = states{16}; % memory containing h and c of the previous timestamp t-1
                    prevmh = mem{1};     % activation layer of the previous timestamp t-1: mean
                    prevSh = mem{2};     % activation layer of the previous timestamp t-1: variance
                    prevmc = mem{3};     % cell c of the previous timestamp t-1: mean   
                    prevSc = mem{4};     % cell c of the previous timestamp t-1: variance

                    % Add noise to Sh_{t-1} and Sc_{t-1} before propagating
                    % to time t (non-stationary hidden and cell states)
%                     if net.LSTMsmoothing == 1  && net.trainMode == 1 
%                         prevSh{j} = prevSh{j} + 1E-3;
%                         prevSc{j} = prevSc{j} + 1E-3;
%                     end

                    if B==1&&rB==1
                        [mz{j}, Sz{j},...
                            mga{j}, Sga{j}, Jga{j},...
                            mc{j}, Sc{j}, Jca{j}, Cch{j}] = tagi.lstmMeanVarB1(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ...
                            [ma{j-1}; prevmh{j}], [Sa{j-1}; prevSh{j}], prevmc{j}, prevSc{j},...
                            nodes(j-1)+nodes(j), nodes(j), B, rB, net.gateActFunIdx, net.RNNbias, net.gpu);
                    else 
                        maBatch1  = reshape(ma{j-1},[nodes(j-1),B]);
                        maBatch2  = reshape(prevmh{j},[nodes(j),B]);
                        maBatch   = reshape([maBatch1; maBatch2],[],1);
                        SaBatch1  = reshape(Sa{j-1},[nodes(j-1),B]);
                        SaBatch2  = reshape(prevSh{j},[nodes(j),B]);
                        SaBatch   = reshape([SaBatch1; SaBatch2],[],1);
                        [mz{j}, Sz{j},...
                            mga{j}, Sga{j}, Jga{j},...
                            mc{j}, Sc{j}, Jca{j}, Cch{j}] = tagi.lstmMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb), ...
                            maBatch, SaBatch, prevmc{j}, prevSc{j},...
                            nodes(j-1)+nodes(j), nodes(j), B, rB, net.gateActFunIdx, net.RNNbias, net.gpu);
                    end
                    
                % GRU
                elseif layer(j) == net.layerEncoder.gru
                    mem    = states{13}; % memory containing h of the previous timestamp t-1
                    prevmh = mem{1};     % activation layer of the previous timestamp t-1: mean
                    prevSh = mem{2};     % activation layer of the previous timestamp t-1: variance
                    if B==1&&rB==1
                        [mz{j}, Sz{j},...
                            mga{j}, Sga{j}, Jga{j}, Jh{j}] = tagi.gruMeanVarB1(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb),...
                                                        ma{j-1}, Sa{j-1}, prevmh{j}, prevSh{j},...
                                                         nodes(j-1)+nodes(j), nodes(j), B, rB, net.RNNbias, net.gpu);
                    else
                        [mz{j}, Sz{j},...
                            mga{j}, Sga{j}, Jga{j}, Jh{j}] = tagi.gruMeanVar(mz{j}, Sz{j}, mw(idxw), Sw(idxw), mb(idxb), Sb(idxb),...
                                                        ma{j-1}, Sa{j-1}, prevmh{j}, prevSh{j},...
                                                         nodes(j-1)+nodes(j), nodes(j), B, rB, net.RNNbias, net.gpu);
                    end    
                end     
                
                % Shortcut connection for residual networks 
                if net.xsc(j)~=0&&(net.filter(net.xsc(j))~=net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j)) 
                    idxXsc = net.xsc(j);
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);                   
                    [mxs{j}, Sxs{j}] = tagi.convMeanVar(mxs{j}, Sxs{j}, mwx(idxwx), Swx(idxwx), mbx(idxbx), Sbx(idxbx), ma{idxXsc}, Sa{idxXsc}, net.idxFmwaXsc(idxXsc, :),...
                        1, filter(idxXsc), imgW(j), imgH(j), filter(j), B, rB, net.paddingXsc(idxXsc), net.gpu);
                    % Save convolutional hidden state before adding x
                    % shortcut
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j}, mxs{j}, Sxs{j});
                elseif net.xsc(j)~=0&&(net.filter(net.xsc(j))==net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j))
                    mxs{j}  = mz{net.xsc(j)};
                    Sxs{j}  = Sz{net.xsc(j)};
                    mdxs{j} = mz{j};
                    Sdxs{j} = Sz{j};
                    [mz{j}, Sz{j}] = arrayfun(@twoPlus, mz{j}, Sz{j}, mxs{j}, Sxs{j});
                end
                % Activation
                if actFunIdx(j)~=0
                    [ma{j}, Sa{j}, J{j}] = act.meanVar(mz{j}, mz{j}, Sz{j}, actFunIdx(j), B, rB, net.gpu);
                else
                    ma{j} = mz{j};
                    Sa{j} = Sz{j};
                    J{j}  = ones(size(mz{j}), 'like', mz{j});                    
                end
            end 
            normStat = tagi.compressNormStat(mra, Sra);
            states   = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
            if strcmp(net.RNNtype,'LSTM_lookback') || strcmp(net.RNNtype,'LSTM_stateful') || strcmp(net.RNNtype,'LSTM_stateless')
                mem{5,1} = Cch;
                states   = tagi.lstmCompressStates(states, mga, Sga, Jga, mc, Sc, Jca, mem);
            elseif strcmp(net.RNNtype,'GRU_lookback') || strcmp(net.RNNtype,'GRU_stateful') || strcmp(net.RNNtype,'GRU_stateless')
                mem{3,1} = Jh;
                states   = tagi.gruCompressStates(states, mga, Sga, Jga, mem);
            end
        end
        
        % Inference 
        function [deltaM, deltaS, deltaMx, deltaSx, deltaMz0, deltaSz0, deltaHm, deltaHs, Chh, Ccc, Cxh] = hiddenStateBackwardPass(net, theta, normStat, states, y, Sy, udIdx, maxIdx)
            % Initialization
            [mw, ~, ~, ~, mwx, ~, ~, ~] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, ~, Sxs] = tagi.extractStates(states);
            if strcmp(net.RNNtype,'LSTM_lookback') || strcmp(net.RNNtype,'LSTM_stateful') || strcmp(net.RNNtype,'LSTM_stateless')
                [mga, Sga, Jga, mc, Sc, Jca, ~] = tagi.lstmExtractStates(states);
            elseif strcmp(net.RNNtype,'GRU_lookback') || strcmp(net.RNNtype,'GRU_stateful') || strcmp(net.RNNtype,'GRU_stateless')
                [mga, Sga, Jga, mem] = tagi.gruExtractStates(states);
            end
            [~, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            stride     = cast(net.stride, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            lHL        = numLayers-1;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaM     = cell(numLayers, 1);
            deltaS     = cell(numLayers, 1);
            deltaMx    = cell(numLayers, 1);
            deltaSx    = cell(numLayers, 1);
            deltaMxs   = cell(numLayers, 1);
            deltaMdxs  = cell(numLayers, 1);
            deltaSxs   = cell(numLayers, 1);
            deltaSdxs  = cell(numLayers, 1); 
            
            deltaHm      = cell(numLayers, 1);
            deltaHs      = cell(numLayers, 1);
            deltaHm{1}   = zeros(nodes(1)*B,1);
            deltaHs{1}   = zeros(nodes(1)*B,1);
            
            % save covariances for smoothing
            Chh = cell(numLayers,1); % cov(h_t,h_{t-1}): covariance between hiddens states of t and t-1
            Ccc = cell(numLayers,1); % cov(c_t,c_{t-1}): covariance between cell states of t and t-1
            Cxh = cell(numLayers,1); % cov(h_t,x_t)

            if net.lastLayerUpdate == 1
                if net.learnSv == 0
                    if net.ny == length(net.sv)
                        sv = net.sv';
                    else
                        sv = repmat(net.sv, [net.ny, 1]);
                    end
                    % Update hidden states for the last hidden layer
                    if isempty(Sy)
                        R = repmat(sv.^2, [net.batchSize, 1]);
                    else
                        R = repmat(sv.^2, [net.batchSize, 1]) + Sy;
                    end
                    Szv = Sa{end} + R;
                    if isempty(udIdx)
                        [deltaMz, deltaSz] = tagi.fowardHiddenStateUpdate(ma{lHL+1}, Szv, J{lHL+1}.*Sz{lHL+1}, y, net.gpu);
                        deltaSz(isnan(y)) = 0;
                        deltaMz(isnan(y)) = 0;
                        deltaHm{end} = deltaMz;
                        deltaHs{end} = deltaSz;
                    else 
                        mzf = ma{end}(udIdx);
                        Szf = J{lHL+1}(udIdx).*Sz{lHL+1}(udIdx);
                        ys  = y(udIdx);
                        Szv = Szv(udIdx);
                        deltaMz = zeros(size(mz{lHL+1}), 'like', mz{lHL+1});
                        deltaSz = zeros(size(Sz{lHL+1}), 'like', Sz{lHL+1});
                        [deltaMz(udIdx), deltaSz(udIdx)] = tagi.fowardHiddenStateUpdate(mzf, Szv, Szf, ys, net.gpu);
                    end
                elseif net.learnSv==1                   
                    if strcmp(net.task, 'regression')&&strcmp(net.noiseType, 'hete')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl, net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl, net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        [deltaMlz, deltaSlz, deltaMv2z, deltaSv2z] = tagi.noiseUpdate4regression(Slz, mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z, net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z, net.nl, net.nv2, B, rB);
                    elseif strcmp(net.task, 'regression')&&strcmp(net.noiseType, 'homo')
                        mv2a = net.sv(1);
                        Sv2a = net.sv(2);
                        mla  = ma{end};
                        Slz  = Sz{end};
                        Sla  = Sa{end};
                        Jl   = J{end};  
                        [deltaMz, deltaSz, deltaMv2z, deltaSv2z] = tagi.homoNoiseUpdate4regression(Slz, mla, Sla, Jl, mv2a, Sv2a, y, net.gpu);
                        net.sv(1) = net.sv(1) + sum(deltaMv2z, 1);
                        net.sv(2) = net.sv(2) + sum(deltaSv2z, 1);                       
                    elseif strcmp(net.task, 'classification')
                        [mla, mv2a] = tagi.detachMeanVar(ma{end}, net.nl, net.nv2, B, rB);
                        [Sla, Sv2a] = tagi.detachMeanVar(Sa{end}, net.nl, net.nv2, B, rB);
                        [Slz, ~]  = tagi.detachMeanVar(Sz{end}, net.nl, net.nv2, B, rB);
                        [Jl, Jv2] = tagi.detachMeanVar(J{end}, net.nl, net.nv2, B, rB);
                        % Activate log(\sigma_v2)
                        [mv2a, Sv2a, Cv2a] = act.expFun(mv2a, Sv2a, net.gpu);
                        
                        deltaMlz  = zeros(size(mla), 'like', mla);
                        deltaSlz  = zeros(size(mla), 'like', mla);
                        deltaMv2z = zeros(size(mla), 'like', mla);
                        deltaSv2z = zeros(size(mla), 'like', mla);
                        [deltaMlz(udIdx), deltaSlz(udIdx), deltaMv2z(udIdx), deltaSv2z(udIdx)] = tagi.noiseUpdate4classification_V2(Slz, mla, Sla, Jl, Jv2, mv2a, Sv2a, Cv2a, y, net.sv, udIdx, net.gpu);
                        deltaMz = tagi.attachMeanVar(deltaMlz, deltaMv2z, net.nl, net.nv2, B, rB);
                        deltaSz = tagi.attachMeanVar(deltaSlz, deltaSv2z, net.nl, net.nv2, B, rB);
                    end                    
                end
            else
                deltaMz = y;
                deltaSz = Sy;
                deltaHm{end} = deltaMz;
                deltaHs{end} = deltaSz;
            end
%             deltaHm{end} = deltaMz;
%             deltaHs{end} = deltaSz;
            sv = net.sv;
            for k = (numLayers-1):-1:1
                if kernelSize(k)==stride(k)||(kernelSize(k)==imgW(k)&&stride(k)==1); overlap = 0; else; overlap = 1; end
                if isempty(mdxs{k+1}); nSz = Sz{k+1}; else; nSz = Sdxs{k+1}; end
                if isempty(mdxs{k}); cSz = Sz{k}; else; cSz = Sdxs{k}; end
                
                cSxs = Sxs{k};
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))~=net.filter(k+1)||net.imgW(net.xsc(k+1))~=net.imgH(k+1))
                    [deltaMx{k+1}, deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1}, deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);  
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    if idxXsc>1                                 
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc}, mwx(idxwx),...
                            net.idxSzzUdXsc{idxXsc}, net.idxFCzwaXsc(idxXsc, :), filter(idxXsc), B, rB, size(net.idxFCzwaXsc{idxXsc, 2}, 1), net.gpu); 
                    end                   
                elseif net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))==net.filter(k+1)||net.imgW(net.xsc(k+1))==net.imgH(k+1))
                    [deltaMx{k+1}, deltaSx{k+1}] = tagi.inovationVector(Sxs{k+1}, deltaMzx, deltaSzx, net.gpu);
                    idxXsc = net.xsc(k+1);
                    if idxXsc>1&&~isempty(Sxs{idxXsc})                      
                        [deltaMxs{idxXsc}, deltaSxs{idxXsc}, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, Sxs{idxXsc}, Sdxs{idxXsc}, J{idxXsc}, [],...
                            [], [], [], [], rB, [], net.gpu);
                    elseif idxXsc>1&&isempty(Sdxs{idxXsc})&&isempty(Sxs{idxXsc}) % First shortcut
                        [~, ~, deltaMdxs{idxXsc}, deltaSdxs{idxXsc}] = tagi.xshortDelta(deltaMx{k+1}, deltaSx{k+1}, [], Sz{idxXsc}, J{idxXsc}, [], [], [], [], [], rB, [], net.gpu);
                    end
                end   
                
                % Innovation vector
                [deltaM{k+1}, deltaS{k+1}] = tagi.inovationVector(nSz, deltaMz, deltaSz, net.gpu);
                
                % Max pooling 
                if layer(k+1) == net.layerEncoder.mp       
                    [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.mpHiddenStateBackwardPass(cSz, cSxs, J{k}, deltaM{k+1}, deltaS{k+1}, maxIdx{k+1}, rB, overlap, net.gpu);
                    
                % Average pooling     
                elseif layer(k+1) == net.layerEncoder.ap 
                    [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.agHiddenStateBackwardPass(cSz, cSxs, J{k}, size(net.idxPooling{k}, 2), deltaM{k+1}, deltaS{k+1},...
                        net.idxSzzUd{k}, imgW(k+1), imgH(k+1), filter(k+1), kernelSize(k), B, rB, overlap, net.gpu);
                    
                % Convolutional     
                elseif layer(k+1) == net.layerEncoder.conv 
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.convHiddenStateBackwardPassB1(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                            net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.convHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                                net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), B, rB, net.gpu);
                        end                       
                    end
                    
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    if k > 1||net.convariateEstm
                        [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.tconvHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1},...
                            net.idxSzzUd{k}, net.idxFCzwa(k, :), imgW(k), imgH(k), filter(k), B, rB, net.gpu);                       
                    end
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln || layer(k+1) == net.layerEncoder.bn                     
                    if k > 1||net.convariateEstm
                        Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                        [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.normHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), Shat, epsilon, deltaM{k+1}, deltaS{k+1},...
                            imgW(k), imgH(k), filter(k), B, rB, layer(k), net.layerEncoder, net.gpu);
                    end 
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.fcHiddenStateBackwardPassB1(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), net.gpu);
                        else
                            [deltaMz, deltaSz, deltaMzx, deltaSzx] = tagi.fcHiddenStateBackwardPass(cSz, cSxs, J{k}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), B, rB, net.gpu);
                        end                                               
                    end  
                % LSTM
                elseif  layer(k+1) == net.layerEncoder.lstm
                        if B==1&&rB==1
                            mem    = states{16}; % memory containing h and c of the previous timestamp t-1
                            prevSh = mem{2};     % variances for h_{t-1}: hidden states
                            prevmc = mem{3};     % means for c_{t-1}: cell states
                            prevSc = mem{4};     % variances for c_{t-1}
                            [deltaMz, deltaSz, Chh{k+1}, Ccc{k+1}, Cxh{k+1}] = tagi.lstmHiddenStateBackwardPassB1(cSz, mga{k+1}, Jga{k+1}, tanh(mc{k+1}), prevmc{k+1}, prevSc{k+1}, prevSh{k+1}, Jca{k+1}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), net.gpu);
                            % Chh = cov(h_t,h_{t-1});
                            % Ccc = cov(c_t,c_{t-1});
                            % Cxh = cov(x_t,h^{1stLayer}_{t});
                        else
                            mem    = states{16}; % memory containing h and c of the previous timestamp t-1
                            prevSh = mem{2};     % variances for h_{t-1}: hidden states
                            prevmc = mem{3};     % means for c_{t-1}: cell states
                            prevSc = mem{4};     % variances for c_{t-1}
                            [deltaMz, deltaSz, Chh{k+1}, Ccc{k+1}, Cxh{k+1}] = tagi.lstmHiddenStateBackwardPass(cSz, mga{k+1}, Jga{k+1}, tanh(mc{k+1}), prevmc{k+1}, prevSc{k+1}, prevSh{k+1}, Jca{k+1}, mw(idxw), deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), B, rB, net.gpu);
                        end
                % GRU    
                elseif layer(k+1) == net.layerEncoder.gru
                    mem    = states{13}; % memory containing h of the previous timestamp t-1
                    prevmh = mem{1};     % activation layer of the previous timestamp t-1: mean
                    if k > 1||net.convariateEstm
                        if B==1&&rB==1
                            [deltaMz, deltaSz] = tagi.gruHiddenStateBackwardPassB1(cSz, deltaM{k+1}, deltaS{k+1}, mw(idxw), prevmh{k+1}, mga{k+1}, Jga{k+1}, nodes(k)+nodes(k+1), nodes(k+1), B, rB, net.gpu);
                        else
                            [deltaMz, deltaSz] = tagi.gruHiddenStateBackwardPass(cSz, deltaM{k+1}, deltaS{k+1}, mw(idxw), prevmh{k+1}, mga{k+1}, Jga{k+1}, nodes(k)+nodes(k+1), nodes(k+1), B, rB, net.gpu);
                        end
                    end
                end

                % deltaHm and deltaHs: are the delta for means and variances to update the activation
                % units; they are needed to estimate the
                % posteriors for states
                if k>1
                    deltaHm{k} = deltaMz; 
                    deltaHs{k} = deltaSz;
                end
                
                % Update hidden states from shortcut
                if ~isempty(deltaMxs{k})&&~isempty(deltaMdxs{k})
                    [deltaMzx, deltaSzx, deltaMz, deltaSz] = arrayfun(@fourPlus, deltaMzx, deltaSzx, deltaMz, deltaSz, deltaMxs{k}, deltaSxs{k}, deltaMdxs{k}, deltaSdxs{k});
                elseif ~isempty(deltaMdxs{k})&&isempty(deltaMxs{k})
                    [deltaMz, deltaSz] = arrayfun(@twoPlus, deltaMz, deltaSz, deltaMdxs{k}, deltaSdxs{k});
                end
            end
            deltaMz0 = deltaMz;
            deltaSz0 = deltaSz;
            
        end
        function deltaTheta = parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx)
            % Initialization
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [~, ~, ma, ~, ~, ~, ~, ~, ~] = tagi.extractStates(states);
            if strcmp(net.RNNtype,'LSTM_lookback') || strcmp(net.RNNtype,'LSTM_stateful') || strcmp(net.RNNtype,'LSTM_stateless')
                [mga, Sga, Jga, mc, Sc, Jca, mem] = tagi.lstmExtractStates(states);
            elseif strcmp(net.RNNtype,'GRU_lookback') || strcmp(net.RNNtype,'GRU_stateful') || strcmp(net.RNNtype,'GRU_stateless')
                [mga, Sga, Jga, mem] = tagi.gruExtractStates(states);
            end
            [mra, Sra] = tagi.extractNormStat(normStat);
            numLayers  = length(net.nodes);
            imgW       = cast(net.imgW, net.dtype);
            imgH       = cast(net.imgH, net.dtype);
            filter     = cast(net.filter, net.dtype);
            kernelSize = cast(net.kernelSize, net.dtype);
            B          = cast(net.batchSize, net.dtype);
            rB         = cast(net.repBatchSize, net.dtype);
            nodes      = cast(net.nodes, net.dtype);
            epsilon    = net.epsilon;
            layer      = net.layer;
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            
            deltaMw    = mw;
            deltaSw    = Sw;
            deltaMb    = mb;
            deltaSb    = Sb;
            deltaMwx   = mwx;
            deltaSwx   = Swx;
            deltaMbx   = mbx;
            deltaSbx   = Sbx;
            for k = (numLayers-1):-1:1
                idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                idxb = (numParamsPerlayer_2(2, k)+1):numParamsPerlayer_2(2, k+1);
                %Shortcut connection for residual network
                if net.xsc(k+1)~=0 && (net.filter(net.xsc(k+1))~=net.filter(k+1)||net.imgW(net.xsc(k+1))~=net.imgH(k+1))
                    idxXsc = net.xsc(k+1); 
                    idxwx = (numParamsPerlayer_2(3, idxXsc)+1):numParamsPerlayer_2(3, idxXsc+1);
                    idxbx = (numParamsPerlayer_2(4, idxXsc)+1):numParamsPerlayer_2(4, idxXsc+1);
                    [deltaMwx(idxwx), deltaSwx(idxwx), deltaMbx(idxbx), deltaSbx(idxbx)] = tagi.convParameterBackwardPass(deltaMwx(idxwx), deltaSwx(idxwx), deltaMbx(idxbx), deltaSbx(idxbx),...
                        Swx(idxwx), Sbx(idxbx), ma{idxXsc}, deltaMx{k+1}, deltaSx{k+1},...
                        net.idxFmwaXsc(idxXsc, :), net.paddingXsc(idxXsc), 1, filter(idxXsc), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu);
                end 
                
                % Convolutional     
                if layer(k+1) == net.layerEncoder.conv  
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.convParameterBackwardPassB1(Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                            net.idxFmwa(k, :), net.padding(k), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.convParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                            net.idxFmwa(k, :), net.padding(k), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu);
                    end                   
                % Transposed convolutional
                elseif layer(k+1) == net.layerEncoder.tconv 
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.tconvParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                        net.idxSwzUd{k}, net.idxFCwz(k, :), kernelSize(k), filter(k), imgW(k+1), imgH(k+1), filter(k+1), B, rB, net.gpu); 
                    
                % Normalization     
                elseif layer(k+1) == net.layerEncoder.ln || layer(k+1) == net.layerEncoder.bn  
                    mhat = tagi.distributeNormMeanVar(mra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                    Shat = tagi.distributeNormMeanVar(Sra{k}, nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), layer(k+1), net.layerEncoder);
                    [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.normParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, mhat, Shat, epsilon, deltaM{k+1}, deltaS{k+1},...
                        nodes(k), imgW(k), imgH(k), filter(k), B, rB, layer(k), net.layerEncoder, net.gpu);        
                    
                % Full-connected     
                elseif  layer(k+1) == net.layerEncoder.fc 
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.fcParameterBackwardPassB1(Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1}, nodes(k), nodes(k+1), net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.fcParameterBackwardPass(deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb), Sw(idxw), Sb(idxb), ma{k}, deltaM{k+1}, deltaS{k+1},...
                            nodes(k), nodes(k+1), B, rB, net.gpu);
                    end
                    
                % LSTM
                elseif  layer(k+1) == net.layerEncoder.lstm
                    mem    = states{16}; % memory containing h and c of previous timestamp t-1
                    prevmh = mem{1};     % activation layer of the last timestamps: mean
                    prevmc = mem{3};     % cell c of the last timestamp: mean
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), ...
                            deltaMb(idxb), deltaSb(idxb)] = tagi.lstmParameterBackwardPassB1(Sw(idxw), Sb(idxb), [ma{k}; prevmh{k+1}], mga{k+1}, Jga{k+1}, tanh(mc{k+1}),...
                                                                    prevmc{k+1}, Jca{k+1}, deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), net.RNNbias, net.gpu);
                    else
                        maBatch1  = reshape(ma{k},[nodes(k),B]);
                        maBatch2  = reshape(prevmh{k+1},[nodes(k+1),B]);
                        maBatch   = reshape([maBatch1; maBatch2],[],1);
                        [deltaMw(idxw), deltaSw(idxw), ...
                            deltaMb(idxb), deltaSb(idxb)] = tagi.lstmParameterBackwardPass(Sw(idxw), Sb(idxb), maBatch, mga{k+1}, Jga{k+1}, tanh(mc{k+1}),...
                                                                    prevmc{k+1}, Jca{k+1}, deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), B, rB, net.RNNbias, net.gpu);
                    end
                 
                % GRU
                elseif  layer(k+1) == net.layerEncoder.gru
                    mem    = states{13}; % memory containing h of previous timestamp t-1
                    prevmh = mem{1};     % activation layer of the last timestamps: mean
                    Jh     = mem{3};
                    if B==1&&rB==1
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.gruParameterBackwardPassB1 (mw(idxw), Sw(idxw), Sb(idxb), ma{k}, ...
                            prevmh{k+1}, mga{k+1}, Jga{k+1}, Jh{k+1}, deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), net.RNNbias,net.gpu);
                    else
                        [deltaMw(idxw), deltaSw(idxw), deltaMb(idxb), deltaSb(idxb)] = tagi.gruParameterBackwardPass (mw(idxw), Sw(idxw), Sb(idxb), ma{k}, ...
                            prevmh{k+1}, mga{k+1}, Jga{k+1}, Jh{k+1}, deltaM{k+1}, deltaS{k+1}, nodes(k)+nodes(k+1), nodes(k+1), B, rB, net.RNNbias, net.gpu);
                    end
                end
            end
            deltaTheta = tagi.compressParameters(deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx);           
        end
        
        % Normalization layer
        function [m, S] = pMeanVar(pm, pS, ni, wi, hi, fi, B, rB, li, lo, le)
            if li == le.fc && lo == le.ln 
                pm = reshape(pm, [ni, B*rB]);
                pS = reshape(pS, [ni, B*rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(ni-1);            
            elseif li == le.fc && lo == le.bn
                pm = reshape(pm, [ni, B*rB]);
                pS = reshape(pS, [ni, B*rB]);
                m  = mean(pm, 2);
                S  = (sum(pS, 2) + sum((pm-m).^2, 2))/(B*rB-1);
            elseif li ~= le.fc && lo == le.ln
                pm = reshape(pm, [wi*hi*fi, B*rB]);
                pS = reshape(pS, [wi*hi*fi, B*rB]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(wi*hi*fi-1);
            elseif li ~= le.fc && lo == le.bn
                pm = reshape(reshape(pm, [wi*hi*fi, B*rB])', [wi*hi*B*rB, fi]);
                pS = reshape(reshape(pS, [wi*hi*fi, B*rB])', [wi*hi*B*rB, fi]);
                m  = mean(pm, 1);
                S  = (sum(pS, 1) + sum((pm-m).^2, 1))/(wi*hi*B*rB - 1);
            end
            m = m(:);
            S = S(:);
        end
        function m = distributeNormMeanVar(m, ni, wi, hi, fi, B, rB, li, lo, le)
            if li == le.fc && lo == le.ln                 
                m  = reshape(repmat(m', [ni, 1]), [ni*B, rB]);                         
            elseif li == le.fc && lo == le.bn
                m  = repmat(m, [B, rB]);
            elseif li ~= le.fc && lo == le.ln
                m  = reshape(repmat(m', [wi*hi*fi, 1]), [wi*hi*fi*B, rB]);
            elseif li ~= le.fc && lo == le.bn
                m  = repmat(reshape(repmat(m', [wi*hi, 1]),[wi*hi*fi, 1]), [B, rB]);
            end
        end
        function [mz, Sz] = fcNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, mhat, Shat, epsilon, B, rB, gpu)
            mb = repmat(mb, [B, 1]);
            Sb = repmat(Sb, [B, 1]);
            mw = repmat(mw, [B, 1]);            
            Sw = repmat(Sw, [B, 1]);                     
            if gpu == 1
                funA = @(x, y) 1./(x+y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat, mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1./(Shat(:, t) + epsilon);
                    mz(:, t) = sqrt(A).*(ma(:, t) - mhat(:, t)).*mw + mb;
                    Sz(:, t) = A.*(Sa(:, t).*(mw.^2) + Sw.*(ma(:, t).^2 - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end
        function [mz, Sz] = convNormMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, mhat, Shat, epsilon,  wi, hi, fi, B, rB, gpu)
            mb   = repmat(reshape(repmat(mb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
            Sb   = repmat(reshape(repmat(Sb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);      
            mw   = repmat(reshape(repmat(mw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);
            Sw   = repmat(reshape(repmat(Sw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);                    
            if gpu == 1
                funA = @(x, y) 1./(x+y);
                A = arrayfun(funA, Shat, epsilon);
                [mz, Sz] = arrayfun(@vectorizedNormMeanVar, ma, Sa, mhat, mw, Sw, mb, Sb, A);
            else
                for t = 1:rB
                    A = 1./(Shat + epsilon);
                    mz(:, t) = sqrt(A).*(ma(:, t) - mhat(:, t)).*mw + mb;
                    Sz(:, t) = A.*(Sa(:, t).*(mw.^2) + Sw.*(ma(:, t).^2 - mhat(:, t).^2 + Sa(:, t))) + Sb;
                end
            end
        end          
        function [deltaMw, deltaSw, deltaMb, deltaSb] = normParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, mra, Sra, epsilon, deltaM, deltaS, ni, wi, hi, fi, B, rB, li, layerEncoder, gpu)
            fun = @(x,y,z,t,q) sqrt((1./(x+q))).*(y-z).*t;
            if li == layerEncoder.fc % Previous layer is full-connected
                Sw = repmat(Sw, [B, 1]);
                Cbz = repmat(Sb, [B, 1]);                 
                if gpu == 1                   
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    % Weights
                    for t = 1:rB 
                        [deltaMwloop, deltaSwloop, deltaMbloop, deltaSbloop] = arrayfun(@vectorizedDelta4normParam, Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                else
                    for t = 1:rB
                        A   = 1./(Sra(:, t) + epsilon);
                        Cwz = sqrt(A).*(ma(:, t) - mra(:, t)).*Sw;
                        deltaMwloop = Cwz.*deltaM(:, t);
                        deltaSwloop = Cwz.*deltaS(:, t).*Cwz;
                        deltaMbloop = Cbz.*deltaM(:, t);
                        deltaSbloop = Cbz.*deltaS(:, t).*Cbz;
                        deltaMw(:, t) = sum(reshape(deltaMwloop, [ni, B]), 2);
                        deltaSw(:, t) = sum(reshape(deltaSwloop, [ni, B]), 2);
                        deltaMb(:, t) = sum(reshape(deltaMbloop, [ni, B]), 2);
                        deltaSb(:, t) = sum(reshape(deltaSbloop, [ni, B]), 2);
                    end
                end
            elseif li == layerEncoder.conv||li == layerEncoder.tconv % Previous layer is convolutional
                Sw = repmat(reshape(repmat(Sw', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
                Cbz = repmat(reshape(repmat(Sb', [wi*hi, 1]), [fi*hi*wi, 1]), [B, 1]);
                if gpu == 1
                    Cwz = arrayfun(fun, Sra, ma, mra, Sw, epsilon);
                    for t = 1:rB                       
                        [deltaMwloop, deltaSwloop, deltaMbloop, deltaSbloop] = arrayfun(@vectorizedDelta4normParam, Cwz(:, t), Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                else
                    for t = 1:rB
                        A   = 1./(Sra(:, t)+epsilon);
                        Cwz = sqrt(A).*(ma(:, t) - mra(:, t)).*Sw;
                        [deltaMwloop, deltaSwloop] = vectorizedDelta(Cwz, deltaM(:, t), deltaS(:, t));
                        [deltaMbloop, deltaSbloop] = vectorizedDelta(Cbz, deltaM(:, t), deltaS(:, t));
                        deltaMwloop = squeeze(permute(reshape(deltaMwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSwloop = squeeze(permute(reshape(deltaSwloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMwloop = sum(sum(deltaMwloop, 1), 2);
                        deltaSwloop = sum(sum(deltaSwloop, 1), 2);
                        deltaMw(:, t) = deltaMwloop(:);
                        deltaSw(:, t) = deltaSwloop(:);
                        % Bias
                        deltaMbloop = squeeze(permute(reshape(deltaMbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaSbloop = squeeze(permute(reshape(deltaSbloop, [wi*hi, 1, fi, B]),[1 2 4 3]));
                        deltaMbloop = sum(sum(deltaMbloop, 1), 2);
                        deltaSbloop = sum(sum(deltaSbloop, 1), 2);
                        deltaMb(:, t) = deltaMbloop(:);
                        deltaSb(:, t) = deltaSbloop(:);
                    end
                end
            end
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMz, deltaSz, deltaMxs, deltaSxs] = normHiddenStateBackwardPass(Sz, Sxs, J, mw, Sra, epsilon, deltaM, deltaS, wi, hi, fi, B, rB, li, layerEncoder, gpu)
            deltaMz = Sz;
            deltaSz = Sz;
            deltaMxs = Sxs;
            deltaSxs = Sxs;
            if li == layerEncoder.fc
                mw = repmat(mw, [B, 1]);                
            elseif li == layerEncoder.conv||li == layerEncoder.tconv
                mw = repmat(reshape(repmat(mw', [wi*hi, 1]), [fi*wi*hi, 1]), [B, 1]);
            end
            if gpu == 1
                if isempty(Sxs)
                    fun = @(x, y, z, t, q) x.*sqrt(1./(y+q)).*z.*t;
                    Czz = arrayfun(fun, J, Sra, Sz, mw, epsilon);
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t)] = arrayfun(@vectorizedDelta, Czz(:, t), deltaM(:, t), deltaS(:, t));
                    end
                else
                    fun = @(x, y, z, q) x.*sqrt(1./(y+q)).*z;
                    Czz = arrayfun(fun, J, Sra, Sz, epsilon);
                    Czx = arrayfun(fun, J, Sra, Sxs, epsilon);
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t), deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw, Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            else
                if isempty(Sxs)
                    A   = 1./(Sra+epsilon);
                    Czz = J.*sqrt(A).*Sz.*mw;
                    for t = 1:rB                      
                        [deltaMz(:, t), deltaSz(:, t)] = vectorizedDelta(Czz, deltaM(:, t), deltaS(:, t));
                    end
                else
                    A   = 1./(Sra+epsilon);
                    Czz = J.*sqrt(A).*Sz;
                    Czx = J.*sqrt(A).*Sz;
                    for t = 1:rB
                        [deltaMz(:, t), deltaSz(:, t), deltaMxs(:, t), deltaSxs(:, t)] = arrayfun(@vectorized4delta, mw, Czz(:, t), Czx(:, t), deltaM(:, t), deltaS(:, t));
                    end
                end
            end
        end
        
        % Full connected layer 
        function [mz, Sz] = fcMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, ni, no, B, rB, gpu)
            idxSum = 1;
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            else
                mb = repmat(mb, [B, 1]);
                Sb = repmat(Sb, [B, 1]);
            end
            mw  = repmat(reshape(mw, [ni, no]), [1, B]);                     
            Sw  = repmat(reshape(Sw, [ni, no]), [1, B]);                                  
            if gpu == 1
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, maloop, mw, Saloop, Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    [mz(:, t), Sz(:, t)] = arrayfun(@twoPlus, mzloop, Szloop, mb, Sb);
                end
            else
                for t = 1:rB
                    maloop = reshape(repmat(reshape(ma(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    Saloop = reshape(repmat(reshape(Sa(:, t), [ni, B]), [no, 1]), [ni, no*B]);
                    [mzloop, Szloop] = vectorizedMeanVar(maloop, mw, Saloop, Sw);
                    mzloop = transpose(sum(mzloop, idxSum));
                    Szloop = transpose(sum(Szloop, idxSum));
                    mz(:, t) = mzloop + mb;
                    Sz(:, t) = Szloop + Sb;
                end
            end            
        end
        function [mz, Sz] = fcMeanVarB1(mw, Sw, mb, Sb, ma, Sa, ni, no, gpu)
            if any(isnan(mb))
                mb = zeros(1,1,'like', mw);
                Sb = zeros(1,1,'like', Sw);               
            end
            mw = reshape(mw, [ni, no]);                     
            Sw = reshape(Sw, [ni, no]); 
            if gpu == 1
                [mzloop, Szloop] = arrayfun(@vectorizedMeanVar, ma, mw, Sa, Sw);
                mzloop = sum(mzloop, 1);
                Szloop = sum(Szloop, 1);
                mzloop = mzloop(:);
                Szloop = Szloop(:);
                [mz, Sz] = arrayfun(@twoPlus, mzloop, Szloop, mb, Sb);
            else
                [mzloop, Szloop] = vectorizedMeanVar(ma, mw, Sa, Sw);
                mzloop = transpose(sum(mzloop, 1));
                Szloop = transpose(sum(Szloop, 1));
                mz = mzloop + mb;
                Sz = Szloop + Sb;
            end            
        end 
        function [deltaMw, deltaSw, deltaMb, deltaSb] = fcParameterBackwardPass(deltaMw, deltaSw, deltaMb, deltaSb, Sw, Sb, ma, deltaMr, deltaSr, ni, no, B, rB, gpu)  
            Cbz = repmat(Sb, [1, B]);
            if gpu == 1  
                for t = 1:rB                   
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)), [ni, 1]),[ni*no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)), [ni, 1]),[ni*no, B]);                  
                    % Weights
%                     Cwz = bsxfun(@times, Sw, maloop);  
                    [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMrw, deltaSrw);
                    deltaMw(:, t) = sum(deltaMrw, 2);
                    deltaSw(:, t) = sum(deltaSrw, 2);
                    % Bias
                    if any(~isnan(Sb))                        
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                      
                        [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMrb, deltaSrb);
                        deltaMb(:, t) = sum(deltaMrb, 2);
                        deltaSb(:, t) = sum(deltaSrb, 2);
                    end
                end
            else
                for t = 1:rB
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]);               
                    deltaMrw = reshape(repmat(transpose(deltaMr(:, t)), [ni, 1]),[ni*no, B]);
                    deltaSrw = reshape(repmat(transpose(deltaSr(:, t)), [ni, 1]),[ni*no, B]); 
                    Cwz      = Sw.*maloop;
                    deltaMrw = Cwz.*deltaMrw;
                    deltaSrw = Cwz.*deltaSrw.*Cwz;
                    deltaMw(:, t) = nansum(deltaMrw, 2);
                    deltaSw(:, t) = nansum(deltaSrw, 2);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(deltaMr(:, t), [no, B]);
                        deltaSrb = reshape(deltaSr(:, t), [no, B]);                        
                        deltaMrb = Cbz.*deltaMrb;
                        deltaSrb = Cbz.*deltaSrb.*Cbz;
                        deltaMb(:, t) = nansum(deltaMrb, 2);
                        deltaSb(:, t) = nansum(deltaSrb, 2);
                    end
                end
            end  
            deltaMw = sum(deltaMw, 2);
            deltaSw = sum(deltaSw, 2);
            deltaMb = sum(deltaMb, 2);
            deltaSb = sum(deltaSb, 2);
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = fcParameterBackwardPassB1(Sw, Sb, ma, deltaMr, deltaSr, ni, no, gpu)  
            Cbz      = Sb;                   
            maloop   = repmat(ma, [no, 1]);
            deltaMrw = repmat(transpose(deltaMr), [ni, 1]);
            deltaMrw = deltaMrw(:);
            deltaSrw = repmat(transpose(deltaSr), [ni, 1]);
            deltaSrw = deltaSrw(:);
            % Weights
            if gpu==1
                [deltaMrw, deltaSrw] = arrayfun(@vectorizedDelta_V2, Sw, maloop, deltaMrw, deltaSrw);
            else
                Cwa = Sw.*maloop;
                deltaMrw = Cwa.*deltaMrw;
                deltaSrw = (Cwa.^2).*deltaSrw;                
            end
            deltaMw = sum(deltaMrw, 2);
            deltaSw = sum(deltaSrw, 2);
            % Bias
            if any(~isnan(Sb))
                if gpu==1
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMr, deltaSr);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMr, deltaSr);
                end
                deltaMb = sum(deltaMrb, 2);
                deltaSb = sum(deltaSrb, 2);
            else
                deltaMb = Sb;
                deltaSb = Sb;
            end
        end
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = fcHiddenStateBackwardPass(Sz, Sxs, J, mw, deltaM, deltaS, ni, no, B, rB, gpu) 
            deltaMz  = Sz;
            deltaSz  = Sz;
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            mw  = repmat(reshape(mw, [ni, no]), [B, 1]);              
            if gpu == 1
                Caz = bsxfun(@times, J, Sz);
                if isempty(Sxs)
                    for t = 1:rB
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Caz(:, t), deltaMzloop, deltaSzloop);
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    Caxs = bsxfun(@times, J, Sxs);
                    for t = 1:rB
                        deltaMloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, mw, Caz(:, t), Caxs(:, t), deltaMloop, deltaSloop);
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            else
                if isempty(Sxs)
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t).*mw;
                        deltaMzloop = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSzloop = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop = Czz.*deltaMzloop;
                        deltaSzloop = Czz.*deltaSzloop.*Czz;
                        deltaMz(:, t) = sum(deltaMzloop, 2);
                        deltaSz(:, t) = sum(deltaSzloop, 2);
                    end
                else
                    for t = 1:rB
                        Czz = J(:, t).*Sz(:, t).*mw;
                        Czx = J(:, t).*Sz(:, t).*mw;
                        deltaMloop     = reshape(repmat(reshape(deltaM(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaSloop     = reshape(repmat(reshape(deltaS(:, t), [no, B]), [ni, 1]), [no, ni*B])';
                        deltaMzloop    = Czz.*deltaMloop;
                        deltaSzloop    = Czz.*deltaSloop.*Czz;
                        deltaMxsloop   = Czx.*deltaMloop;
                        deltaSxsloop   = Czx.*deltaSloop.*Czx;
                        deltaMz(:, t)  = sum(deltaMzloop, 2);
                        deltaSz(:, t)  = sum(deltaSzloop, 2);
                        deltaMzx(:, t) = sum(deltaMxsloop, 2);
                        deltaSzx(:, t) = sum(deltaSxsloop, 2);
                    end
                end
            end
        end 
        function [deltaMz, deltaSz, deltaMzx, deltaSzx] = fcHiddenStateBackwardPassB1(Sz, Sxs, J, mw, deltaM, deltaS, ni, no, gpu) 
            mw  = reshape(mw, [ni, no]);              
            deltaMzx = Sxs;
            deltaSzx = Sxs;
            if isempty(Sxs)
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu==1
                    Caz = bsxfun(@times, J, Sz);
                    [deltaMzloop, deltaSzloop] = arrayfun(@vectorizedDelta_V2, mw, Caz, deltaMloop, deltaSloop);
                else
                    Caz = J.*Sz;
                    Cwa = mw.*Caz;
                    deltaMzloop = Cwa.*deltaMloop;
                    deltaSzloop = (Cwa.^2).*deltaSloop;
                end
                deltaMz = sum(deltaMzloop, 2);
                deltaSz = sum(deltaSzloop, 2);
            else                
                deltaMloop = repmat(deltaM', [ni, 1]);
                deltaSloop = repmat(deltaS', [ni, 1]);
                if gpu==1
                    Caz = bsxfun(@times, J, Sz);
                    Caxs = bsxfun(@times, J, Sxs);
                    [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = arrayfun(@vectorized4delta, mw, Caz, Caxs, deltaMloop, deltaSloop);
                else
                    Caz = J.*Sz;
                    Caxs = J.*Sxs;
                    [deltaMzloop, deltaSzloop, deltaMxsloop, deltaSxsloop] = vectorized4delta(mw, Caz, Caxs, deltaMloop, deltaSloop);
                end
                deltaMz  = sum(deltaMzloop, 2);
                deltaSz  = sum(deltaSzloop, 2);
                deltaMzx = sum(deltaMxsloop, 2);
                deltaSzx = sum(deltaSxsloop, 2);
            end
        end 
        
        % LSTM 
        % batch=1
        function [mz, Sz, mga, Sga, Jga, mc, Sc, Jca, Cch] = lstmMeanVarB1(mz, Sz, mw, Sw, mb, Sb, ma, Sa, prevmc, prevSc, ni, no, B, rB, actFunIdx, RNNbias, gpu)            
            if RNNbias == 0
                mb  = zeros(size(mb), 'like', mw);
                Sb  = zeros(size(Sb), 'like', Sw);
            end
            [mg, Sg] = tagi.fcMeanVar(repmat(mz,[4,1]), repmat(Sz,[4,1]), mw, Sw, mb, Sb, ma, Sa, ni, 4*no, B, rB, gpu);
            mg = reshape(mg,[],4);
            Sg = reshape(Sg,[],4);
            [mga, Sga, Jga] = act.meanVar(mg, mg, Sg, actFunIdx(1), B, rB, gpu);
            [mga(:,3), Sga(:,3), Jga(:,3)] = act.meanVar(mg(:,3), mg(:,3), Sg(:,3), actFunIdx(3), B, rB, gpu);
           
            [mc1, Sc1] = vectorizedMeanVar (mga(:, 1), prevmc, Sga(:, 1), prevSc);
            [mc2, Sc2] = vectorizedMeanVar(mga(:, 2), mga(:, 3), Sga(:, 2),  Sga(:, 3));
            
            mc = mc1 + mc2;
            Sc = Sc1 + Sc2;
            [mca, Sca, Jca] = act.meanVar(mc, mc, Sc, 1, B, rB, gpu);

            Cch = Jca.*Sc.*mga(:,4); % cov(ct,ht)
            [mz, Sz] = vectorizedMeanVar(mga(:, 4), mca, Sga(:, 4), Sca);
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = lstmParameterBackwardPassB1(Sw, Sb, ma, mga, Jga, mca, prevmca, Jca, deltaMr, deltaSr, ni, no, RNNbias, gpu)              
            % clear code
%             Jf   = Jga(:, 1);
%             Ji   = Jga(:, 2);
%             Jct  = Jga(:, 3);
%             Jo   = Jga(:, 4);           
%             mi  = mga(:, 2);
%             mct = mga(:, 3);
%             mo  = mga(:, 4);        
%             deltaMrf  = Jca.*Jf.*prevmca.*mo;           
%             deltaMri  = Jca.*Ji.*mct.*mo;         
%             deltaMrct = Jca.*Jct.*mi.*mo;           
%             deltaMro  = Jo.*mca;          
%             deltaMrg  = [deltaMrf, deltaMri, deltaMrct, deltaMro]; 

            if RNNbias == 0
                Sb  = zeros(size(Sb), 'like', Sw);
            end
            % fast code
            deltaMrf  = Jca.*Jga(:,1).*prevmca.*mga(:,4);
            deltaMri  = Jca.*Jga(:,2).*mga(:,3).*mga(:,4);
            deltaMrct = Jca.*Jga(:,3).*mga(:,2).*mga(:,4);
            deltaMro  = Jga(:,4).*mca;
            deltaMrg  = [deltaMrf, deltaMri, deltaMrct, deltaMro];
            
            [deltaMw, deltaSw, deltaMb, deltaSb] = tagi.fcParameterBackwardPassB1_V2(Sw, Sb, ma, repmat(deltaMr,[4,1]), repmat(deltaSr,[4,1]), deltaMrg(:), ni, 4*no, gpu); 
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb, Cwa] = fcParameterBackwardPassB1_V2(Sw, Sb, ma, deltaMr, deltaSr, deltaM, ni, no, gpu)
            Cbz      = Sb;                   
            maloop   = repmat(ma, [no, 1]);
            deltaMrw = repmat(transpose(deltaMr), [ni, 1]);
            deltaMrw = deltaMrw(:);
            deltaSrw = repmat(transpose(deltaSr), [ni, 1]);
            deltaSrw = deltaSrw(:);
            deltaMref = deltaM;
            deltaM   = repmat(transpose(deltaM), [ni, 1]);  
            deltaM   = deltaM(:);
                       
            % Weights
            Cwa = Sw.*deltaM.*maloop;
            deltaMrw = Cwa.*deltaMrw;
            deltaSrw = (Cwa.^2).*deltaSrw;
            deltaMw = sum(deltaMrw, 2);
            deltaSw = sum(deltaSrw, 2);
            % Bias
            if any(~isnan(Sb))
                Cbz = Cbz.*deltaMref;
                if gpu==1
                    [deltaMrb, deltaSrb] = arrayfun(@vectorizedDelta, Cbz, deltaMr, deltaSr);
                else
                    [deltaMrb, deltaSrb] = vectorizedDelta(Cbz, deltaMr, deltaSr);
                end
                deltaMb = sum(deltaMrb, 2);
                deltaSb = sum(deltaSrb, 2);
            else
                deltaMb = Sb;
                deltaSb = Sb;
            end
        end
        function [deltaMz, deltaSz, Chh, Ccc, Cxh] = lstmHiddenStateBackwardPassB1(Sz, mga, Jg, mca, prevmc, prevSc, prevSh, Jca, mw, deltaM, deltaS, ni, no, gpu)
            Sz = [Sz;prevSh];
            mw  = reshape(mw, [ni*no, 4]); 
            
            mwf = reshape(mw(:, 1), [ni, no]);
            mwi = reshape(mw(:, 2), [ni, no]);        
            mwc = reshape(mw(:, 3), [ni, no]);
            mwo = reshape(mw(:, 4), [ni, no]);

            deltaMloop = repmat(deltaM', [ni, 1]);
            deltaSloop = repmat(deltaS', [ni, 1]);
            
            Czzf = Jg(:,1)'.*Sz.*mwf.*prevmc';
            Czzi = Jg(:,2)'.*Sz.*mwi.*mga(:,3)';
            Czzc = Jg(:,3)'.*Sz.*mwc.*mga(:,2)';
            Czzo = Jg(:,4)'.*Sz.*mwo.*mca';
            Czz  = Czzo + Jca'.*(Czzf + Czzi + Czzc).*mga(:,4)';
                     
            deltaMzloop = Czz.*deltaMloop;
            deltaSzloop = (Czz.^2).*deltaSloop;
            deltaMz = sum(deltaMzloop, 2);
            deltaSz = sum(deltaSzloop, 2);
            deltaMz = deltaMz(1:end-no,:);
            deltaSz = deltaSz(1:end-no,:);

            Chh = Czz(end-no+1:end,:);   % cov(h_t,h_{t-1})
            Ccc = prevSc.*mga(:,1);      % cov(c_t,c_{t-1})
            Cxh = Czz(1:end-no,:);       % cov(x_t,h_t)
        end 
        function [Chh, Ccc, Cxh] = lstmCov4smoother(Sz, mga, Jg, mca, prevmc, prevSc, prevSh, Jca, mw, ni, no)
            Sz = [Sz;prevSh];
            mw  = reshape(mw, [ni*no, 4]);

            mwf = reshape(mw(:, 1), [ni, no]);
            mwi = reshape(mw(:, 2), [ni, no]);
            mwc = reshape(mw(:, 3), [ni, no]);
            mwo = reshape(mw(:, 4), [ni, no]);

            Czzf = Jg(:,1)'.*Sz.*mwf.*prevmc';
            Czzi = Jg(:,2)'.*Sz.*mwi.*mga(:,3)';
            Czzc = Jg(:,3)'.*Sz.*mwc.*mga(:,2)';
            Czzo = Jg(:,4)'.*Sz.*mwo.*mca';
            Czz  = Czzo + Jca'.*(Czzf + Czzi + Czzc).*mga(:,4)';

            Chh     = Czz(end-no+1:end,:);   % cov(h_t,h_{t-1})
            Ccc     = prevSc.*mga(:,1);      % cov(c_t,c_{t-1})
            Cxh     = Czz(1:end-no,:);       % cov(x_t,h_t)
        end
        % batch>1
        function [mz, Sz, mga, Sga, Jga, mc, Sc, Jca, Cch] = lstmMeanVar(mz, Sz, mw, Sw, mb, Sb, ma, Sa, prevmc, prevSc, ni, no, B, rB, actFunIdx, RNNbias, gpu)
            if RNNbias == 0
                mb  = zeros(size(mb), 'like', mw);
                Sb  = zeros(size(Sb), 'like', Sw);
            end
            [mg, Sg] = tagi.fcMeanVar(repmat(mz,[4,1]), repmat(Sz,[4,1]), mw, Sw, mb, Sb, ma, Sa, ni, 4*no, B, rB, gpu);
            mg = reshape(reshape(mg,[],B),no,4,[]);
            mg = reshape(permute(mg,[1,3,2]),[],4);
            Sg = reshape(reshape(Sg,[],B),no,4,[]);
            Sg = reshape(permute(Sg,[1,3,2]),[],4);
            
            [mga, Sga, Jga] = act.meanVar(mg, mg, Sg, actFunIdx(1), B, rB, gpu);
            [mga(:,3), Sga(:,3), Jga(:,3)] = act.meanVar(mg(:,3), mg(:,3), Sg(:,3), actFunIdx(3), B, rB, gpu);

            [mc1, Sc1] = vectorizedMeanVar (mga(:, 1), prevmc, Sga(:, 1), prevSc);
            [mc2, Sc2] = vectorizedMeanVar(mga(:, 2), mga(:, 3), Sga(:, 2),  Sga(:, 3));

            mc = mc1 + mc2;
            Sc = Sc1 + Sc2;
            [mca, Sca, Jca] = act.meanVar(mc, mc, Sc, 1, B, rB, gpu);

            Cch = Jca.*Sc.*mga(:,4); % cov(ct,ht)
            [mz, Sz] = vectorizedMeanVar(mga(:, 4), mca, Sga(:, 4), Sca);

        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = lstmParameterBackwardPass(Sw, Sb, ma, mga, Jga, mca, prevmca, Jca, deltaMr, deltaSr, ni, no, B, rB, RNNbias, gpu)          
            % clear code
%             Jf   = Jga(:, 1);
%             Ji   = Jga(:, 2);
%             Jct  = Jga(:, 3);
%             Jo   = Jga(:, 4);           
%             mi  = mga(:, 2);
%             mct = mga(:, 3);
%             mo  = mga(:, 4);        
%             deltaMrf  = Jca.*Jf.*prevmca.*mo;           
%             deltaMri  = Jca.*Ji.*mct.*mo;         
%             deltaMrct = Jca.*Jct.*mi.*mo;           
%             deltaMro  = Jo.*mca;          
%             deltaMrg  = [deltaMrf, deltaMri, deltaMrct, deltaMro];  

            if RNNbias == 0
                Sb  = zeros(size(Sb), 'like', Sw);
            end
            % code
            deltaMrf  = Jca.*Jga(:, 1).*prevmca.*mga(:, 4);
            deltaMri  = Jca.*Jga(:, 2).*mga(:, 3).*mga(:, 4);
            deltaMrct = Jca.*Jga(:, 3).*mga(:, 2).*mga(:, 4);
            deltaMro  = Jga(:, 4).*mca;
            deltaMrg  = [deltaMrf, deltaMri, deltaMrct, deltaMro];
            [deltaMw, deltaSw, deltaMb, deltaSb] = tagi.fcParameterBackwardPass_V2(Sw, Sb, ma, repmat(deltaMr,[4,1]), repmat(deltaSr,[4,1]), deltaMrg(:), ni, 4*no, B, rB, gpu); 
        end
        function [deltaMw, deltaSw, deltaMb, deltaSb] = fcParameterBackwardPass_V2(Sw, Sb, ma, deltaMr, deltaSr, deltaM, ni, no, B, rB, gpu)              
            deltaM = reshape(permute(reshape(deltaM,no/4,B,[]),[1,3,2]),[],B);
            Cbz       = Sb.*deltaM;
            if gpu == 1  
                for t = 1:rB
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]); 
                    deltaMrw = reshape(permute(reshape(deltaMr,no/4,B,[]),[1,3,2]),[],B);
                    deltaMrw = repelem(deltaMrw,ni,1);
                    deltaSrw = reshape(permute(reshape(deltaSr,no/4,B,[]),[1,3,2]),[],B);
                    deltaSrw = repelem(deltaSrw,ni,1);
                    deltaM   = repelem(deltaM,ni,1);
                                       
                    Cwz      = Sw.*deltaM.*maloop;
                    deltaMrw = Cwz.*deltaMrw;
                    deltaSrw = Cwz.*deltaSrw.*Cwz;
                    deltaMw(:, t) = nansum(deltaMrw, 2);
                    deltaSw(:, t) = nansum(deltaSrw, 2);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(permute(reshape(deltaMr,no/4,B,[]),[1,3,2]),[],B);
                        deltaSrb = reshape(permute(reshape(deltaSr,no/4,B,[]),[1,3,2]),[],B);
                        deltaMrb = Cbz.*deltaMrb;
                        deltaSrb = Cbz.*deltaSrb.*Cbz;
                        deltaMb(:, t) = nansum(deltaMrb, 2);
                        deltaSb(:, t) = nansum(deltaSrb, 2);
                    end
                end
            else
                for t = 1:rB
                    maloop   = repmat(reshape(ma(:, t), [ni, B]), [no, 1]); 
                    deltaMrw = reshape(permute(reshape(deltaMr,no/4,B,[]),[1,3,2]),[],B);
                    deltaMrw = repelem(deltaMrw,ni,1);
                    deltaSrw = reshape(permute(reshape(deltaSr,no/4,B,[]),[1,3,2]),[],B);
                    deltaSrw = repelem(deltaSrw,ni,1);
                    deltaM   = repelem(deltaM,ni,1);
                    
                    Cwz      = Sw.*deltaM.*maloop;
                    deltaMrw = Cwz.*deltaMrw;
                    deltaSrw = Cwz.*deltaSrw.*Cwz;
                    deltaMw(:, t) = nansum(deltaMrw, 2);
                    deltaSw(:, t) = nansum(deltaSrw, 2);
                    if any(~isnan(Sb))
                        deltaMrb = reshape(permute(reshape(deltaMr,no/4,B,[]),[1,3,2]),[],B);
                        deltaSrb = reshape(permute(reshape(deltaSr,no/4,B,[]),[1,3,2]),[],B);
                        deltaMrb = Cbz.*deltaMrb;
                        deltaSrb = Cbz.*deltaSrb.*Cbz;
                        deltaMb(:, t) = nansum(deltaMrb, 2);
                        deltaSb(:, t) = nansum(deltaSrb, 2);
                    end
                end
            end  
        end
        function [deltaMz, deltaSz, Chh, Ccc, Cxh] = lstmHiddenStateBackwardPass(Sz, mga, Jg, mca, prevmca, prevSc, prevSh, Jca, mw, deltaM, deltaS, ni, no, B, rB, gpu)                                                   
            niR = ni;
            mw   = reshape(mw,ni,no,4);
            mw   = mw(1:niR,:,:);
            mw   = repmat(mw,B,1,1);
            mwF  = mw(:,:,1);
            mwI  = mw(:,:,2);
            mwCt = mw(:,:,3);
            mwO  = mw(:,:,4);

            Sz = reshape(Sz,[],B);
            prevSh = reshape(prevSh,[],B);
            Sz = [Sz;prevSh];
            Sz = reshape(Sz,[],1);

            deltaMloop = reshape(permute(reshape(deltaM,1,no,[]),[1,3,2]),B,[]);
            deltaMloop = repelem(deltaMloop,niR,1);
            deltaSloop = reshape(permute(reshape(deltaS,1,no,[]),[1,3,2]),B,[]);
            deltaSloop = repelem(deltaSloop,niR,1);

            Czzf = repelem(reshape(Jg(:,1),[],B)',niR,1).*Sz.*mwF.*repelem(reshape(prevmca,[],B)',niR,1);
            Czzi = repelem(reshape(Jg(:,2),[],B)',niR,1).*Sz.*mwI.*repelem(reshape(mga(:,3),[],B)',niR,1);
            Czzc = repelem(reshape(Jg(:,3),[],B)',niR,1).*Sz.*mwCt.*repelem(reshape(mga(:,2),[],B)',niR,1);
            Czzo = repelem(reshape(Jg(:,4),[],B)',niR,1).*Sz.*mwO.*repelem(reshape(mca,[],B)',niR,1);

            Czz = Czzo + repelem(reshape(Jca,[],B)',niR,1).*(Czzf + Czzi + Czzc).*repelem(reshape(mga(:,4),[],B)',niR,1);

            deltaMzloop = Czz.*deltaMloop;
            deltaSzloop = (Czz.^2).*deltaSloop;
            deltaMz = sum(deltaMzloop, 2);
            deltaSz = sum(deltaSzloop, 2);

            deltaMz = reshape(deltaMz,[],B);
            deltaSz = reshape(deltaSz,[],B);
            deltaMz = deltaMz(1:end-no,:);
            deltaSz = deltaSz(1:end-no,:);
            deltaMz = reshape(deltaMz,[],1);
            deltaSz = reshape(deltaSz,[],1);

            idxx = repmat([1:ni-no]',[1 B])+[0:B-1]*ni; 
            idxh = repmat([ni-no+1:ni]',[1 B])+[0:B-1]*ni; 
            Chh     = Czz(idxh,:);          % cov(h_t,h_{t-1})
            Ccc     = prevSc.*mga(:,1);     % cov(c_t,c_{t-1})
            Cxh     = Czz(idxx,:);          % cov(x_t,h_t)
        end 
        %
        function [states] = lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, Cch)
            % Update LSTM's hidden states
            states{3} = cellfun(@plus, states{3}, deltaHm, 'Uniform', 0);
            states{4} = cellfun(@plus, states{4}, deltaHs, 'Uniform', 0);
            % Update LSTM's cell states
            deltaCm   = cellfun(@times, Cch, deltaM,'UniformOutput',false);
            deltaCs   = cellfun(@(x1,x2) x1.^2.*x2, Cch, deltaS,'UniformOutput',false);
            states{13} = cellfun(@plus, states{13}, deltaCm, 'Uniform', 0);
            states{14} = cellfun(@plus, states{14}, deltaCs, 'Uniform', 0);
        end
        function x = dataMasking (x, nbtest)
            x(end-nbtest+1:end,:) = nan;
        end
        function states = lstmCompressStates(states, mag, Sag, Jg, mc, Sc, Jc, mem)
            % LSTM's states
            states{10} = mag;
            states{11} = Sag;
            states{12} = Jg;
            states{13} = mc;
            states{14} = Sc;
            states{15} = Jc;
            states{16} = mem;
        end
        function [mga, Sga, Jga, mc, Sc, Jca, mem] = lstmExtractStates(states)
            % LSTM's states
            mga = states{10};
            Sga = states{11};
            Jga = states{12};
            mc  = states{13};
            Sc  = states{14};
            Jca = states{15};
            mem = states{16};
        end
        function [mem] = updateRnnMemory (RNNtype, states)
            if strcmp(RNNtype,'LSTM_lookback') || strcmp(RNNtype,'LSTM_stateful') || strcmp(RNNtype,'LSTM_stateless')
                mem = cell(4,1);
                mem{1} = states{3};
                mem{2} = states{4};
                mem{3} = states{13};
                mem{4} = states{14};
            elseif strcmp(RNNtype,'GRU_lookback') || strcmp(RNNtype,'GRU_stateful') || strcmp(RNNtype,'GRU_stateless')
                mem = cell(2,1);
                mem{1} = states{3};
                mem{2} = states{4};
            end
        end
        function states = lstmInitializeStates(states)
            % Additional variables for LSTM
            mag = states{1};
            Sag = states{1};
            Jg  = states{1};
            mc  = states{1};
            Sc  = states{1};
            Jc  = states{1};   
            mem = [];
            states = tagi.lstmCompressStates(states, mag, Sag, Jg, mc, Sc, Jc, mem);
        end
        function states = lstmInitializeInputs(states, mz0, Sz0, ma0, Sa0, J0, mdxs0, Sdxs0, mxs0, Sxs0, xsc, mem)
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            [mag, Sag, Jg, mc, Sc, Jc] = tagi.lstmExtractStates(states);
            % Normal net
            mz{1} = mz0;
            if any(isempty(Sz0))
                Sz{1} = zeros(size(mz0), 'like', mz0);
            else
                Sz{1} = Sz0;
            end
            if any(isempty(ma0))
                ma{1} = mz0;
            else
                ma{1} = ma0;
            end 
            if any(isempty(Sa0))
                Sa{1} = Sz{1};
            else
                Sa{1} = Sa0;
            end   
            if any(isempty(J0))
                J{1} = ones(size(mz0), 'like', mz0);
            else
                J{1} = J0;
            end  
            % Residual net
            if any(isempty(mdxs0))&&~all(xsc==0)
                mdxs{1} = mz0;
            else
                mdxs{1} = mdxs0;
            end
            if any(isempty(Sdxs0))&&~all(xsc==0)
                Sdxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sdxs{1} = Sdxs0;
            end
            if any(isempty(mxs0))&&~all(xsc==0)
                mxs{1} = mz0;
            else
                mxs{1} = mxs0;
            end
            if any(isempty(Sxs0))&&~all(xsc==0)
                Sxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sxs{1} = Sxs0;
            end
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
            states = tagi.lstmCompressStates(states, mag, Sag, Jg, mc, Sc, Jc, mem);
        end
        
        % Shared functions for update step
        function [deltaM, deltaS] = inovationVector(SzF, dMz, dSz, gpu)
            if gpu == 1
                iSzF  = bsxfun(@rdivide, 1, SzF);
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS] = arrayfun(@vectorizedDelta, iSzF, dMz, dSz);
            else              
                iSzF   = 1./SzF; 
                iSzF(isinf(iSzF)) = zeros(1,1, 'like', dMz);
                [deltaM, deltaS]  = vectorizedDelta(iSzF, dMz, dSz);
            end           
        end 
        function [deltaMz, deltaSz] = fowardHiddenStateUpdate(mzF, SzF, Cyz, y, gpu)
            if gpu == 1
                dz  = y - mzF;
                SzF = 1./SzF;
                SzF(isinf(SzF)) = 0;
                K = bsxfun(@times, Cyz, SzF);
                deltaMz = bsxfun(@times, K, dz);
                deltaSz = bsxfun(@times, -K, Cyz);
            else
                dz  = y - mzF;
                SzF = 1./SzF;
                SzF(isinf(SzF)) = 0;
                K = Cyz.*SzF;
                deltaMz = K.*dz;
                deltaSz = -K.*Cyz;
            end
        end   
        function theta = globalParameterUpdate(theta, deltaTheta, gpu)          
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx] = tagi.extractParameters(deltaTheta);
            if gpu==1
                [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            else
                [mw, Sw]   = twoPlus(mw, Sw, deltaMw, deltaSw);
                [mb, Sb]   = twoPlus(mb, Sb, deltaMb, deltaSb);
                [mwx, Swx] = twoPlus(mwx, Swx, deltaMwx, deltaSwx);
                [mbx, Sbx] = twoPlus(mbx, Sbx, deltaMbx, deltaSbx);
            end
            theta      = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        function theta = globalParameterUpdateMultiGPUs(theta, deltaTheta, numParamsPerlayer, numDevices)  
            numParams  = sum(numParamsPerlayer, 2);           
            deltaTheta = cat(2, deltaTheta{:});
            [deltaMw, deltaSw, deltaMb, deltaSb, deltaMwx, deltaSwx, deltaMbx, deltaSbx] = tagi.extractParameters_V2(deltaTheta);
            deltaMw  = cat(1, deltaMw{:});
            deltaSw  = cat(1, deltaSw{:});
            deltaMb  = cat(1, deltaMb{:});
            deltaSb  = cat(1, deltaSb{:});
            deltaMwx = cat(1, deltaMwx{:});
            deltaSwx = cat(1, deltaSwx{:});
            deltaMbx = cat(1, deltaMbx{:});
            deltaSbx = cat(1, deltaSbx{:});  
            
            deltaMw  = sum(reshape(cat(1, deltaMw{:}), [numParams(1), numDevices]), 2);
            deltaSw  = sum(reshape(cat(1, deltaSw{:}), [numParams(1), numDevices]), 2);
            deltaMb  = sum(reshape(cat(1, deltaMb{:}), [numParams(2), numDevices]), 2);
            deltaSb  = sum(reshape(cat(1, deltaSb{:}), [numParams(2), numDevices]), 2);
            deltaMwx = sum(reshape(cat(1, deltaMwx{:}), [numParams(3), numDevices]), 2);
            deltaSwx = sum(reshape(cat(1, deltaSwx{:}), [numParams(3), numDevices]), 2);
            deltaMbx = sum(reshape(cat(1, deltaMbx{:}), [numParams(4), numDevices]), 2);
            deltaSbx = sum(reshape(cat(1, deltaSbx{:}), [numParams(4), numDevices]), 2);            
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.extractParameters(theta);
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);            
            [mw, Sw]   = arrayfun(@twoPlus, mw, Sw, deltaMw, deltaSw);
            [mb, Sb]   = arrayfun(@twoPlus, mb, Sb, deltaMb, deltaSb);
            [mwx, Swx] = arrayfun(@twoPlus, mwx, Swx, deltaMwx, deltaSwx);
            [mbx, Sbx] = arrayfun(@twoPlus, mbx, Sbx, deltaMbx, deltaSbx);
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.distributeParameters2Layers(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx, numParamsPerlayer);
            theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
        end
        
        % AGVI
        function [deltaMz, deltaSz] = noiseBackwardUpdate(maF, SaF, CzzF, maB, SaB, gpu)
            if gpu == 1
                funM    = @(x, y, z) x.*(y-z);
                funS    = @(x, y, z) x.*(y-z).*x;
                Jz      = CzzF./SaF;
                deltaMz = arrayfun(funM, Jz, maB, maF);
                deltaSz = arrayfun(funS, Jz, SaB, SaF);
            else
                Jz      = CzzF./SaF;
                deltaMz = Jz.*(maB - maF);
                deltaSz = Jz.*(SaB - SaF).*Jz;
            end
        end
        
        % Initialization for weights and bias   
        function theta = initializeWeightBias(net)
%             rng(1223)
            %Initialization
            nodes     = double(net.nodes);
            numLayers = length(net.nodes);
            layer     = net.layer;
            idxw      = net.idxw;
            idxwXsc   = net.idxwXsc;
            idxbXsc   = net.idxbXsc;
            idxb      = net.idxb;
            biasStd   = 1E-2;
            B         = net.batchSize;
            rB        = net.repBatchSize;
            noParam   = nan;
            gainM     = cast(net.gainM, net.dtype);
            gainS     = cast(net.gainS, net.dtype);  
%             gainM     = gainS;
            mw        = tagi.createInitCellwithArray(numLayers-1);
            Sw        = tagi.createInitCellwithArray(numLayers-1);
            mb        = tagi.createInitCellwithArray(numLayers-1);
            Sb        = tagi.createInitCellwithArray(numLayers-1);
            mwx       = tagi.createInitCellwithArray(numLayers-1);
            Swx       = tagi.createInitCellwithArray(numLayers-1);
            mbx       = tagi.createInitCellwithArray(numLayers-1);
            Sbx       = tagi.createInitCellwithArray(numLayers-1);
            for j = 2:numLayers
                if ~isempty(idxw{j-1})                    
                    if layer(j) == net.layerEncoder.conv || layer(j) == net.layerEncoder.tconv % Conv. layer
                        fanIn  = (cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j-1), net.dtype);
                        if net.xsc(j-1)~=0
                            fanIn = 2*fanIn;
                        end
                        if strcmp(net.initParamType, 'Xavier')
                            if j<numLayers&&(layer(j+1) == net.layerEncoder.mp || layer(j+1) == net.layerEncoder.ap)
                                fanOut= ((cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j), net.dtype))/(cast(net.kernelSize(j), net.dtype).^2);
                            else
                                fanOut= ((cast(net.kernelSize(j-1), net.dtype).^2)*cast(net.filter(j), net.dtype));
                            end
                            Sw{j-1} = (gainS(j-1))*(2/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);
                        elseif strcmp(net.initParamType, 'He')
                            Sw{j-1} = (gainS(j-1))*(1/(fanIn))*ones(length(idxw{j-1}), 1, net.dtype);
                        end 
                        mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                        if ~isempty(idxb{j-1})
                            Sb{j-1} = (1/fanIn)*ones(length(idxb{j-1}), 1, net.dtype);
                            mb{j-1} = randn(length(Sb{j-1}), 1).*sqrt(Sb{j-1});
                        end
                    elseif layer(j) == net.layerEncoder.ln || layer(j) == net.layerEncoder.bn
                        Sb{j-1} = 1E-4*gainS(j-1)*ones(length(idxb{j-1}), 1, net.dtype);
                        mb{j-1} = 0*rand(length(Sb{j-1}), 1, net.dtype).*sqrt(Sb{j-1});
                        Sw{j-1} = 1*ones(length(idxw{j-1}), 1, net.dtype);
                        mw{j-1} = 1*ones(length(idxw{j-1}), 1, net.dtype);
                    else
                        if layer(j) == 7 || layer(j) == 8
                            fanIn  = nodes(j-1) + nodes(j);
                        else
                            fanIn  = nodes(j-1);
                        end
                        fanOut = nodes(j);
                        if strcmp(net.initParamType, 'Xavier')
                            Sw{j-1} = (gainS(j-1))*(1/(fanIn+fanOut))*ones(length(idxw{j-1}), 1, net.dtype);
                        elseif strcmp(net.initParamType, 'He')
                            Sw{j-1} = (gainS(j-1))*(1/(fanIn))*ones(length(idxw{j-1}), 1, net.dtype);
                        end
                        mw{j-1} = gainM(j-1)*randn(length(Sw{j-1}), 1).*sqrt(Sw{j-1});
                        if ~isempty(idxb{j-1})
                            Sb{j-1} = (0.1/(fanIn+fanOut))*ones(length(idxb{j-1}), 1, net.dtype);
                            mb{j-1} = randn(length(Sb{j-1}), 1).*sqrt(Sb{j-1});
                        end
                    end  
                else
                    mw{j-1} = noParam;
                    Sw{j-1} = noParam; 
                    Sb{j-1} = noParam;
                    mb{j-1} = noParam;
                end 
                if net.xsc(j)~=0&&(net.filter(net.xsc(j))~=net.filter(j)||net.imgW(net.xsc(j))~=net.imgW(j))
                    idxXsc = net.xsc(j);                                     
                    fanIn  = cast(net.filter(idxXsc), net.dtype);
                    fanOut = cast(net.filter(j), net.dtype);
                    if strcmp(net.initParamType, 'Xavier')
                        Swx{idxXsc} = (gainS(idxXsc))*(2/(fanIn+fanOut))*ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    elseif strcmp(net.initParamType, 'He')
                        Swx{idxXsc} = (1/(fanIn))*ones(length(idxwXsc{idxXsc}), 1, net.dtype);
                    end
                    mwx{idxXsc} = randn(length(Swx{idxXsc}), 1).*sqrt(Swx{idxXsc});
                    if ~isempty(idxbXsc{idxXsc})
                        Sbx{idxXsc} = 1E-6*ones(length(idxbXsc{idxXsc}), 1, net.dtype);
                        mbx{idxXsc} = 0*randn(length(Sbx{idxXsc}), 1).*sqrt(Sbx{idxXsc});
                    end                   
                    if net.gpu == 1
                        mwx{idxXsc} = gpuArray(mwx{idxXsc});
                        Swx{idxXsc} = gpuArray(Swx{idxXsc});
                        mbx{idxXsc} = gpuArray(mbx{idxXsc});
                        Sbx{idxXsc} = gpuArray(Sbx{idxXsc});
                    end
                end
                clear fanIn
                % Send to gpu
                if net.gpu == 1
                    mw{j-1} = gpuArray(mw{j-1});
                    Sw{j-1} = gpuArray(Sw{j-1});
                    mb{j-1} = gpuArray(mb{j-1});
                    Sb{j-1} = gpuArray(Sb{j-1});                    
                end
            end 
            [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = tagi.catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx);
           theta = tagi.compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx); 
        end    
        function states = initializeStates(nodes, B, rB, xsc, dtype, gpu)
            % Normal net
            numLayers = length(nodes);          
            mz  = tagi.createStateCellarray(nodes, numLayers, B, rB, dtype, gpu); 
            Sz  = mz; 
            ma  = mz;
            Sa  = mz;
            J   = mz;
            % Residual net
            idx = xsc~=0;
            mdxs = cell(numLayers, 1);
            mdxs(idx) = mz(idx);
            Sdxs = mdxs;
            mxs  = mdxs;
            Sxs  = mdxs;
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function [deltaxs, deltadxs] = initializeShortcutStateDelta(xsc, idxXsc, x, B, rB)
            layers   = xsc(xsc~=0);
            deltaxs  = cell(length(xsc), 1);
            deltadxs = cell(length(xsc), 1);
            for j = layers
                if ~isempty(idxXsc{j})
                    deltaxs{j}  = zeros(length(idxXsc{j})*B, rB, 'like', x{j});
                    deltadxs{j} = deltaxs{j};
                else
                    deltadxs{j} = zeros(size(x{j}), 'like', x{j});
                    deltaxs{j}  = zeros(size(x{j}), 'like', x{j});
                end
            end
        end
        function states = initializeInputs(states, mz0, Sz0, ma0, Sa0, J0, mdxs0, Sdxs0, mxs0, Sxs0, xsc)
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = tagi.extractStates(states);
            % Normal net
            mz{1} = mz0;
            if any(isempty(Sz0))
                Sz{1} = zeros(size(mz0), 'like', mz0);
            else
                Sz{1} = Sz0;
            end
            if any(isempty(ma0))
                ma{1} = mz0;
            else
                ma{1} = ma0;
            end 
            if any(isempty(Sa0))
                Sa{1} = Sz{1};
            else
                Sa{1} = Sa0;
            end   
            if any(isempty(J0))
                J{1} = ones(size(mz0), 'like', mz0);
            else
                J{1} = J0;
            end  
            % Residual net
            if any(isempty(mdxs0))&&~all(xsc==0)
                mdxs{1} = mz0;
            else
                mdxs{1} = mdxs0;
            end
            if any(isempty(Sdxs0))&&~all(xsc==0)
                Sdxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sdxs{1} = Sdxs0;
            end
            if any(isempty(mxs0))&&~all(xsc==0)
                mxs{1} = mz0;
            else
                mxs{1} = mxs0;
            end
            if any(isempty(Sxs0))&&~all(xsc==0)
                Sxs{1} = zeros(size(mz0), 'like', mz0);
            else
                Sxs{1} = Sxs0;
            end
            states = tagi.compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs);
        end
        function maxIdx = initializeMaxPoolingIndices(nodes, layers, layerEncoder, B, rB, dtype, gpu)
            if gpu==1
                zeroPad = zeros(1, 1, dtype, 'gpuArray');
            else
                zeroPad = zeros(1, 1, dtype);
            end
            numLayers = length(nodes);
            maxIdx = cell(numLayers, 1);
            maxPoolingLayers = find(layers==layerEncoder.mp);
            if ~isempty(maxPoolingLayers)
                for j = maxPoolingLayers
                    maxIdx{j} = zeros(nodes(j)*B, rB, 'like', zeroPad);
                end
            end
        end
        function normStat = initializeNormStat(nodes, filter, B, rB, layers, layerEncoder, x)
            numLayers = length(nodes);
            mra = cell(numLayers, 1);
            layNorm = layers==layerEncoder.ln;
            batNormConv = layers==layerEncoder.bn&(layers==layerEncoder.conv|layers==layerEncoder.tconv|layers==layerEncoder.mp|layers==layerEncoder.ap);
            batNormfc = layers==layerEncoder.bn&layers==layerEncoder.fc;
            for j = layNorm
                mra{j} = zeros(B, rB, 'like', x);
            end
            for j = batNormfc
                mra{j} = zeros(nodes(j), rB, 'like', x);
            end
            for j = batNormConv
                mra{j} = zeros(filter(j), rB, 'like', x);
            end
            Sra = mra;
            normStat = tagi.compressNormStat(mra, Sra);
        end  
        function deltaTheta = initializeDeltaTheta(theta, rB, numLayers)
            deltaTheta = cell(numLayers-1, 1);
            for j = 1:numLayers-1
                deltaTheta{j} = repmat(theta{j}, [1, rB]);
            end
        end
        function mw = fcOrthWeights(mw, Sw, ni, no)
            M = reshape(mw, [ni, no])';
            [r,c] = size(M);
            if r == c
                [~,~,W] = svd(M);
                mw = reshape(W', [ni*no, 1]);
            elseif r > c
                N = M(1:c,:);
                [~,~,W] = svd(N);
                W  = [W;M(c+1:end,:)];
                mw = reshape(W', [ni*no, 1]);
            else
                d = c-r;
                D = randn(d,c)*sqrt(Sw(1));
                M =[M;D];
                [~,~,W] = svd(M);
                W  = W(1:r,:);
                mw = reshape(W', [ni*no, 1]);
            end          
        end
        function mw = convOrthWeights(mw, Sw, ki, fi, fo)
            M = reshape(mw, [ki*ki*fi, fo])';
            [r,c] = size(M);
            if r == c
                [~,~,W] = svd(M);
                 mw = reshape(W', [ki*ki*fi*fo, 1]);
            elseif r > c
                N = M(1:c,:);
                [~,~,W] = svd(N);
                W  = [W;M(c+1:end,:)];
                mw = reshape(W', [ki*ki*fi*fo, 1]);
            else
                d = c-r;
                D = randn(d,c)*sqrt(Sw(1));
                M =[M;D];
                [~,~,W] = svd(M);
                W  = W(1:r,:);
                mw = reshape(W', [ki*ki*fi*fo, 1]);
            end            
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = catParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
            mw  = cat(1, mw{:});
            Sw  = cat(1, Sw{:});
            mb  = cat(1, mb{:});
            Sb  = cat(1, Sb{:});
            mwx = cat(1, mwx{:});
            Swx = cat(1, Swx{:});
            mbx = cat(1, mbx{:});
            Sbx = cat(1, Sbx{:});
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = distributeParameters2Layers(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx, numParams)
            mw  = mat2cell(mw, numParams(1, :));
            Sw  = mat2cell(Sw, numParams(1, :));
            mb  = mat2cell(mb, numParams(2, :));
            Sb  = mat2cell(Sb, numParams(2, :));
            mwx = mat2cell(mwx, numParams(3, :));
            Swx = mat2cell(Swx, numParams(3, :));
            mbx = mat2cell(mbx, numParams(4, :));
            Sbx = mat2cell(Sbx, numParams(4, :));
        end    
       
        % Storing
        function states = compressStates(mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs)
            states = cell(9, 1);
            states{1} = mz;
            states{2} = Sz;
            states{3} = ma;
            states{4} = Sa;
            states{5} = J;
            states{6} = mdxs;
            states{7} = Sdxs;
            states{8} = mxs;
            states{9} = Sxs;
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = extractStates(states)
            mz   = states{1};
            Sz   = states{2};
            ma   = states{3};
            Sa   = states{4};
            J    = states{5};
            mdxs = states{6};
            Sdxs = states{7};
            mxs  = states{8};
            Sxs  = states{9};
        end
        function [mz, Sz, ma, Sa, J, mdxs, Sdxs, mxs, Sxs] = extractStatesMultiGPUs(states)
            spmd
                mz   = states{1};
                Sz   = states{2};
                ma   = states{3};
                Sa   = states{4};
                J    = states{5};
                mdxs = states{6};
                Sdxs = states{7};
                mxs  = states{8};
                Sxs  = states{9};
            end
        end
        function theta = compressParameters(mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx)
            theta     = cell(8, 1);
            theta{1}  = mw;
            theta{2}  = Sw;
            theta{3}  = mb;
            theta{4}  = Sb;
            theta{5}  = mwx;
            theta{6}  = Swx;
            theta{7}  = mbx;
            theta{8}  = Sbx;
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = extractParameters(theta)
            mw  = theta{1};
            Sw  = theta{2};
            mb  = theta{3};
            Sb  = theta{4};
            mwx = theta{5};
            Swx = theta{6};
            mbx = theta{7};
            Sbx = theta{8};
        end
        function [mw, Sw, mb, Sb, mwx, Swx, mbx, Sbx] = extractParameters_V2(theta)
            mw  = theta(1, :);
            Sw  = theta(2, :);
            mb  = theta(3, :);
            Sb  = theta(4, :);
            mwx = theta(5, :);
            Swx = theta(6, :);
            mbx = theta(7, :);
            Sbx = theta(8, :);
        end
        function normStat = compressNormStat(mra, Sra)
            normStat = cell(2, 1);
            normStat{1} = mra;
            normStat{2} = Sra;
        end
        function [mra, Sra] = extractNormStat(normStat)
            mra = normStat{1};
            Sra = normStat{2};
        end  
       
        % Create cell with an array
        function x = createInitCellwithArray(numLayers)
            x = cell(numLayers, 1);
            x(:) = {nan};
        end
        function z = createStateCellarray(nodes, numLayers, B, rB, dtype, gpu)   
            z = cell(numLayers, 1);
            if gpu == 1
                zeroPad = zeros(1,1,dtype, 'gpuArray');
            else
                zeroPad = zeros(1,1,dtype);
            end
            for j = 2:numLayers               
                z{j} = zeros(nodes(j)*B, rB, 'like', zeroPad);
            end
        end                       
        function normStat = createInitNormStat(net)
            mra    = cell(length(net.nodes) -1, 1);            
            Sra    = cell(length(net.nodes) -1, 1);
            if net.gpu == 1
                mra(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
                Sra(:) = {zeros(1, 1, net.dtype, 'gpuArray')};
            else
                mra(:) = {zeros(1, 1, net.dtype)};
                Sra(:) = {zeros(1, 1, net.dtype)};
            end
            normStat = tagi.compressNormStat(mra, Sra);
        end
        
        % Batches of input for RNN
        function [x, y, nb_del] = prepDataBatch_RNN (x, y, batchSize, sql)            
            if batchSize == 1
                x = reshape(x,[],1,size(x,2));
                nb_del = 0;
            else 
                nbCov = size(x,2);
                batchSize = double(batchSize);
                pad = rem(size(y,1)-sql,batchSize);
                nb_del = pad;
                nbobs = (size(y,1)-nb_del-sql)/batchSize;
                y(1:nb_del,:) = [];
                x(1:nb_del,:) = [];
                idxEnd = sql;
                for i = 1:batchSize
                    idxEnd   = idxEnd+nbobs;
                    y_(:,i) = y(idxEnd-sql-nbobs+1:idxEnd,:);
                end
                y = y_;

                for i = 1:size(x,2)
                    idxEnd = sql;
                    for j = 1:batchSize
                        idxEnd   = idxEnd+nbobs;
                        x_(:,j,i) = x(idxEnd-sql-nbobs+1:idxEnd,i);
                    end
                end
                x = x_;
            end
        end

        % Smoothing for LSTM
        function [rnnMem0, zOsm, SzOsm, Sq_infer, rnnMemory] = rnnSmoother(net, rnnSmooth)
            layer = net.layer;
            nblayer = numel(layer);

            % Load before smoothing: cell, hidden states, and z^{O}:
            % priors, posteriors, covariances
            cPrior = rnnSmooth.cPrior;      % cell states: Prior
            ScPrior = rnnSmooth.ScPrior;
            hPrior = rnnSmooth.hPrior;      % hidden states: Prior
            ShPrior = rnnSmooth.ShPrior;
            cPos = rnnSmooth.cPos;          % cell states: Posterior
            ScPos = rnnSmooth.ScPos;
            hPos  = rnnSmooth.hPos;         % hidden states: Posterior
            ShPos = rnnSmooth.ShPos;
            Cxh = rnnSmooth.Cxh;            % cov(x_t,h_t) input and hidden states of 1st hidden layer
            Ccc   = rnnSmooth.Ccc;      % cov(c_t,c_{t-1})
            Chh   = rnnSmooth.Chh;      % cov(h_t,h_{t-1})
            Czz   = rnnSmooth.Czz;      % cov(z^{O}_t,z^{O}_{t-1})

            % to save smoothed values
            c0Sm   = cell(nblayer,1);   % cell states: smoother values
            Sc0Sm  = cell(nblayer,1);
            h0Sm   = cell(nblayer,1);   % hidden states: smoother values
            Sh0Sm  = cell(nblayer,1);
            h1Sm   = cell(nblayer,1);
            Sh1Sm  = cell(nblayer,1);
            hSm   = cell(nblayer,1);
            ShSm  = cell(nblayer,1);
            cSm   = cell(nblayer,1);
            ScSm  = cell(nblayer,1);
            initMem = rnnSmooth.initMem;

            % only do smoothing for LSTM, layer(i) = 7
            for i = 1:nblayer
                if layer(i)==7
                    cPrior_   = cell2mat(cPrior(i,:));
                    ScPrior_  = cell2mat(ScPrior(i,:));
                    hPrior_   = cell2mat(hPrior(i,:));
                    ShPrior_  = cell2mat(ShPrior(i,:));
                    cPos_   = cell2mat(cPos(i,:));
                    ScPos_  = cell2mat(ScPos(i,:));
                    hPos_   = cell2mat(hPos(i,:));
                    ShPos_  = cell2mat(ShPos(i,:));
                    Ccc_ = cell2mat(Ccc(i,:));
                    Chh_ = cell2mat(permute(Chh(i,:),[1,3,2]));
                    % smoothing for cell states
                    [cSm{i}, ScSm{i}] = tagi.KFSmootherCell_batch(cPrior_, ScPrior_, cPos_, ScPos_, Ccc_, net.batchSize);
                    % smoothing for hidden states
                    [hSm{i}, ShSm{i}] = tagi.KFSmootherHidden_batch(hPrior_, ShPrior_, hPos_, ShPos_, Chh_, net.batchSize);
                end
            end

            % smoothing for z^{O}
            [zOsm, SzOsm] = tagi.KFSmootherZ0(rnnSmooth);

            % save the smoothed estimates for cell, hidden states at all
            % time steps
            rnnMemory{1,1} = hSm;
            rnnMemory{2,1} = ShSm;
            rnnMemory{3,1} = cSm;
            rnnMemory{4,1} = ScSm;
            [rnnMemory] = tagi.RnnSplitMemoryBatch1(net.layer, rnnMemory, size(hSm{2},2));

            % smoothed estimates for cell, hidden states to be used as c0
            % and h0 at the next epoch. 
            rnnMem0 = rnnMemory(:,net.nb_past_infer+1);
            % The sequence length used for the next epoch
            Sq_infer{1} = zOsm(1:net.nb_past_infer);
            Sq_infer{2} = SzOsm(1:net.nb_past_infer);
        end

        function [xsm, Sxsm] = KFSmootherCell_batch(xpre, Sxpre, xup, Sxup, cov, B)
            nb = size(xpre,1)/B;
            xsm  = zeros(size(xpre),'single');
            Sxsm = zeros(size(Sxpre),'single');
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            for i = size(xpre,2)-1:-1:1
                for j = 1:B
                    idx = [1:nb]' + nb*(j-1);
                    J = cov(idx,i+1)./Sxpre(idx,i+1);
                    xsm(idx,i)  = xup(idx,i) + J.*(xsm(idx,i+1)-xpre(idx,i+1));
                    Sxsm(idx,i) = Sxup(idx,i) + (J.^2).*(Sxsm(idx,i+1)-Sxpre(idx,i+1));
                end
            end
        end
        function [xsm, Sxsm] = KFSmootherHidden_batch(xpre, Sxpre, xup, Sxup, cov, B)
            nb = size(xpre,1)/B;
            xsm  = zeros(size(xpre),'single');
            Sxsm = zeros(size(Sxpre),'single');
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            for i = size(xpre,2)-1:-1:1
                for j = 1:B
                    idx = [1:nb]' + nb*(j-1);
                    J = cov(idx,:,i+1)./Sxpre(idx,i+1)';
                    xsm(idx,i)  = xup(idx,i)+ sum(J.*(xsm(idx,i+1)-xpre(idx,i+1))',2);
                    Sxsm(idx,i) = Sxup(idx,i) + sum((J.^2).*(Sxsm(idx,i+1)-Sxpre(idx,i+1))',2);
                end
            end
        end
        function [zOsm, SzOsm] = KFSmootherZ0(rnnSmooth)
            Czz = rnnSmooth.Czz;
            zOPos = rnnSmooth.zOPos;
            SzOPos = rnnSmooth.SzOPos;
            zOPrior = rnnSmooth.zOPrior;
            SzOPrior = rnnSmooth.SzOPrior;
            [zOsm, SzOsm]  = tagi.KFSmootherHidden_v1(zOPrior, SzOPrior, zOPos, SzOPos, Czz);
        end
        function [cov_zz_output] = covZZlstm(net, theta, ma, mem, Chh)
            mh = ma{end-1};
            prevmh = mem{1};
            prevmh = prevmh{end-1};
            numLayers  = length(net.nodes);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            [mw, Sw, mb, Sb] = tagi.extractParameters(theta);
            idxw = (numParamsPerlayer_2(1, numLayers-1)+1):numParamsPerlayer_2(1, numLayers);
            idxb = (numParamsPerlayer_2(2, numLayers-1)+1):numParamsPerlayer_2(2, numLayers);
            mw = mw(idxw);
            Sw = Sw(idxw);
            Sb = Sb(idxb);
            cov_zz_output = sum(Sw.*diag(Chh),'all') + sum(Sw.*mh.*prevmh,'all') + sum(Chh.*(mw*mw'),'all') + Sb; % cov(z^{(O)}_t,z^{(O)}_{t-1}): cov of output of lstm between t and t-1
        end
        function [cov_zz_output] = covZZlstm_2outputs(net, theta, ma, mem, Chh)
            mh = ma{end-1};
            prevmh = mem{1};
            prevmh = prevmh{end-1};
            numLayers  = length(net.nodes);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            [mw, Sw, mb, Sb] = tagi.extractParameters(theta);
            idxw = (numParamsPerlayer_2(1, numLayers-1)+1):numParamsPerlayer_2(1, numLayers);
            idxb = (numParamsPerlayer_2(2, numLayers-1)+1):numParamsPerlayer_2(2, numLayers);
            idxw = idxw(1:length(idxw)/2);
            idxb = idxb(1:length(idxb)/2);
            mw = mw(idxw);
            Sw = Sw(idxw);
            Sb = Sb(idxb);
            cov_zz_output = sum(Sw.*diag(Chh),'all') + sum(Sw.*mh.*prevmh,'all') + sum(Chh.*(mw*mw'),'all') + Sb; % cov(z^{(O)}_t,z^{(O)}_{t-1}): cov of output of lstm between t and t-1
        end

        function [xsm, Sxsm] = KFSmootherHidden_v1(xpre, Sxpre, xup, Sxup, cov)
            xsm  = zeros(size(xpre),'single');
            Sxsm = zeros(size(Sxpre),'single');
            xsm(:,end) = xup(:,end);
            Sxsm(:,end) = Sxup(:,end);
            for i = size(xpre,2)-1:-1:1
                J = cov(:,:,i+1)./Sxpre(:,i+1)';
                xsm(:,i)  = xup(:,i)+ sum(J.*(xsm(:,i+1)-xpre(:,i+1))',2);
                Sxsm(:,i) = Sxup(:,i) + sum((J.^2).*(Sxsm(:,i+1)-Sxpre(:,i+1))',2);
            end
        end
        
        function [zOsm, SzOsm] = KFSmootherInput(xPos, SxPos, hPrior, ShPrior, hSm, ShSm, cov)
            hSm  = hSm{2};
            ShSm = ShSm{2};
            hSm(:,1) = []; 
            ShSm(:,1) = []; 
            hPrior   = cell2mat(hPrior(2,2:end));
            ShPrior  = cell2mat(ShPrior(2,2:end));
            cov = cov(2,:);
            zOsm  = zeros(size(xPos),'single');
            SzOsm = zeros(size(SxPos),'single');
            for i = size(hSm,2):-1:1
                cov_temp = cov{i};
                cov_temp = cov_temp(end,:);
                J = cov_temp./ShPrior(:,i)';
                zOsm(:,i)  = xPos(:,i)+ sum(J.*(hSm(:,i)-hPrior(:,i))',2);
                SzOsm(:,i) = SxPos(:,i) + sum((J.^2).*(ShSm(:,i)-ShPrior(:,i))',2);
            end
            zOsm = zOsm';
            SzOsm = SzOsm';
        end
        function [Sq0] = KFSmootherSq(net, xPos, SxPos, hPrior, ShPrior, hSm, ShSm, cov)
            hSm  = hSm{2};
            ShSm = ShSm{2};
            hSm = hSm(:,2);
            ShSm = ShSm(:,2);
            hPrior   = cell2mat(hPrior(2,2));
            ShPrior  = cell2mat(ShPrior(2,2));
            cov = cov(2,1);
            zOsm  = zeros(size(xPos),'single');
            SzOsm = zeros(size(SxPos),'single');
            for i = size(hSm,2):-1:1
                cov_temp = cov{i};
                cov_temp = cov_temp(end,:);
                J = cov_temp./ShPrior(:,i)';
                zOsm(:,i)  = xPos(:,i)+ sum(J.*(hSm(:,i)-hPrior(:,i))',2);
                SzOsm(:,i) = SxPos(:,i) + sum((J.^2).*(ShSm(:,i)-ShPrior(:,i))',2);
            end
            Sq0{1} = zOsm(net.nbCov+1:end);
            Sq0{2} = SzOsm(net.nbCov+1:end);
        end

        function [rnnMem_split] = RnnSplitMemoryBatch1(layer, rnnMem, batchSize)
            rnnMem_split_ = cell(size(rnnMem{1},1), batchSize);
            rnnMem_split = cell(4,batchSize);
            for i = 1:size(rnnMem,1)
                mem_ = rnnMem{i};
                for j = 1:size(mem_,1)
                    if layer(j) == 7 || layer(j) == 8
                        mem = reshape(mem_{j},[],batchSize);
                        node = size(mem,1);
                        mem = mat2cell(mem,node,ones(1,batchSize));
                    else
                        mem = cell(1,batchSize);
                    end
                    rnnMem_split_(j,:) = mem;
                end
                for k = 1:batchSize
                    rnnMem_split(i,k) = {rnnMem_split_(:,k)};
                end
            end
        end

        function [Chh, Ccc, Cxh] = cov4smoother(net, theta, states)
            numLayers  = length(net.nodes);
            numParamsPerlayer_2 = net.numParamsPerlayer_2;
            layer = net.layer;
            nodes = cast(net.nodes, net.dtype);
            [mw, Sw, mb, Sb] = tagi.extractParameters(theta);
            [mz, Sz, ma, Sa, J, mdxs, Sdxs, ~, Sxs] = tagi.extractStates(states);
            if strcmp(net.RNNtype,'LSTM_lookback') || strcmp(net.RNNtype,'LSTM_stateful') || strcmp(net.RNNtype,'LSTM_stateless')
                [mga, Sga, Jga, mc, Sc, Jca, ~] = tagi.lstmExtractStates(states);
            elseif strcmp(net.RNNtype,'GRU_lookback') || strcmp(net.RNNtype,'GRU_stateful') || strcmp(net.RNNtype,'GRU_stateless')
                [mga, Sga, Jga, mem] = tagi.gruExtractStates(states);
            end
            Chh = cell(numLayers,1); % cov(h_t,h_{t-1}): covariance between hiddens states of t and t-1
            Ccc = cell(numLayers,1); % cov(c_t,c_{t-1}): covariance between cell states of t and t-1
            Cxh = cell(numLayers,1); % cov(h_t,x_t)
            for k = (numLayers-1):-1:1
                if  layer(k+1) == net.layerEncoder.lstm
                    idxw = (numParamsPerlayer_2(1, k)+1):numParamsPerlayer_2(1, k+1);
                    cSz = Sz{k};
                    mem    = states{16}; % memory containing h and c of the previous timestamp t-1
                    prevSh = mem{2};     % activation layer of the previous timestamp t-1: variance
                    prevmc = mem{3};     % cell c of the previous timestamp t-1: mean
                    prevSc = mem{4};     % cell c of the previous timestamp t-1: variance
                    [Chh{k+1}, Ccc{k+1}, Cxh{k+1}] = tagi.lstmCov4smoother(cSz, mga{k+1}, Jga{k+1}, tanh(mc{k+1}), prevmc{k+1}, prevSc{k+1}, prevSh{k+1}, Jca{k+1}, mw(idxw), nodes(k)+nodes(k+1), nodes(k+1));
                end
            end
        end
    end
end