%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         resnet
% Description:  Residual networks
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      May 23, 2020
% Updated:      September 2, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef resnet
    methods (Static)
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = v1(imgSize, ny, numBlock)
            % Input
            layer0          = [2                2       6];
            filter0         = [imgSize(3)       16      16];
            kernelSize0     = [3                1       3];
            padding0        = [1                0       1];
            paddingType0    = [1                0       1];
            stride0         = [1                0       1];
            nodes0          = [prod(imgSize)    0       0];
            actFunIdx0      = [0                4       0];
            connect0        = [0                0       0];
            xsc0            = [0                0       1];
            % Stage 1
            layer1          = [2    6   2   6];
            filter1         = [16   16  16  16];
            kernelSize1     = [1    3   1   3];
            padding1        = [0    1   0   1];
            paddingType1    = [0    1   0   1];
            stride1         = [0    1   0   1];
            actFunIdx1      = [4    0   4   0];
            numBlocks1      = numBlock;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    6   2   6];
            filter2         = [32   32  32  32];
            kernelSize2     = [1    3   1   3];
            padding2        = [0    1   0   1];
            paddingType2    = [0    1   0   1];
            stride2         = [0    1   0   1];
            actFunIdx2      = [4    0   4   0];
            numBlocks2      = numBlock;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    6   2   6];
            filter3         = [64   64  64  64];
            kernelSize3     = [1    3   1   3];
            padding3        = [0    1   0   1];
            paddingType3    = [0    1   0   1];
            stride3         = [0    1   0   1];
            actFunIdx3      = [4    0   4   0];
            numBlocks3      = numBlock;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 8;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [64   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = D14(imgSize, ny)
            % Input
            layer0          = 2;
            filter0         = imgSize(3);
            kernelSize0     = 3;
            padding0        = 1;
            paddingType0    = 1;
            stride0         = 1;
            nodes0          = prod(imgSize);
            actFunIdx0      = 0;
            connect0        = 0;
            xsc0            = 1;
            % Stage 1
            layer1          = [2    6   2   6];
            filter1         = [16   16  16  16]/2;
            kernelSize1     = [1    3   1   3];
            padding1        = [0    1   0   1];
            paddingType1    = [0    1   0   1];
            stride1         = [0    1   0   1];
            actFunIdx1      = [4    0   4   0];
            numBlocks1      = 1;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    6   2   6];
            filter2         = [32   32  32  32]/2;
            kernelSize2     = [1    3   1   3];
            padding2        = [0    1   0   1];
            paddingType2    = [0    1   0   1];
            stride2         = [0    1   0   1];
            actFunIdx2      = [4    0   4   0];
            numBlocks2      = 1;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    6   2   6];
            filter3         = [64   64  64  64]/2;
            kernelSize3     = [1    3   1   3];
            padding3        = [0    1   0   1];
            paddingType3    = [0    1   0   1];
            stride3         = [0    1   0   1];
            actFunIdx3      = [4    0   4   0];
            numBlocks3      = 1;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 8;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [32   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = D20Keras(imgSize, ny)
            % Input
            layer0          = [2                2       6];
            filter0         = [imgSize(3)       16      16];
            kernelSize0     = [3                1       3];
            padding0        = [1                0       1];
            paddingType0    = [1                0       1];
            stride0         = [1                0       1];
            nodes0          = [prod(imgSize)    0       0];
            actFunIdx0      = [0                0       4];
            connect0        = [0                0       0];
            xsc0            = [0                0       1];
            % Stage 1
            layer1          = [2    6   2   6];
            filter1         = [16   16  16  16];
            kernelSize1     = [1    3   1   3];
            padding1        = [0    1   0   1];
            paddingType1    = [0    1   0   1];
            stride1         = [0    1   0   1];
            actFunIdx1      = [0    4   0   4];
            numBlocks1      = 3;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    6   2   6];
            filter2         = [32   32  32  32];
            kernelSize2     = [1    3   1   3];
            padding2        = [0    1   0   1];
            paddingType2    = [0    1   0   1];
            stride2         = [0    1   0   1];
            actFunIdx2      = [0    4   0   4];
            numBlocks2      = 3;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    6   2   6];
            filter3         = [64   64  64  64];
            kernelSize3     = [1    3   1   3];
            padding3        = [0    1   0   1];
            paddingType3    = [0    1   0   1];
            stride3         = [0    1   0   1];
            actFunIdx3      = [0    4   0   4];
            numBlocks3      = 3;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 8;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [64   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = D20Keras_V2(imgSize, ny)
            % Input
            layer0          = [2                2       6];
            filter0         = [imgSize(3)       16      16];
            kernelSize0     = [3                1       3];
            padding0        = [1                0       1];
            paddingType0    = [1                0       1];
            stride0         = [1                0       1];
            nodes0          = [prod(imgSize)    0       0];
            actFunIdx0      = [0                0       4];
            connect0        = [0                0       0];
            xsc0            = [0                0       1];
            % Stage 1
            layer1          = [2    6   2   6];
            filter1         = [16   16  16  16];
            kernelSize1     = [1    3   1   3];
            padding1        = [0    1   0   1];
            paddingType1    = [0    1   0   1];
            stride1         = [0    1   0   1];
            actFunIdx1      = [0    4   0   4];
            numBlocks1      = 3;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    6   2   6];
            filter2         = [32   32  32  32];
            kernelSize2     = [1    3   1   3];
            padding2        = [0    1   0   1];
            paddingType2    = [0    1   0   1];
            stride2         = [0    1   0   1];
            actFunIdx2      = [0    4   0   4];
            numBlocks2      = 3;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    6   2   6];
            filter3         = [64   64  64  64];
            kernelSize3     = [1    3   1   3];
            padding3        = [0    1   0   1];
            paddingType3    = [0    1   0   1];
            stride3         = [0    1   0   1];
            actFunIdx3      = [0    4   0   4];
            numBlocks3      = 3;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 8;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [64   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = D20Keras_V3(imgSize, ny)
            % Input
            layer0          = [2                2       6];
            filter0         = [imgSize(3)       16      16];
            kernelSize0     = [3                1       3];
            padding0        = [1                0       1];
            paddingType0    = [1                0       1];
            stride0         = [1                0       1];
            nodes0          = [prod(imgSize)    0       0];
            actFunIdx0      = [0                4       0];
            connect0        = [0                0       0];
            xsc0            = [0                0       1];
            % Stage 1
            layer1          = [2    2   6];
            filter1         = [16   16  16];
            kernelSize1     = [3    1   3];
            padding1        = [1    0   1];
            paddingType1    = [1    0   1];
            stride1         = [1    0   1];
            actFunIdx1      = [4    4   0];
            numBlocks1      = 3;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    2   6];
            filter2         = [32   32  32];
            kernelSize2     = [3    1   3];
            padding2        = [1    0   1];
            paddingType2    = [1    0   1];
            stride2         = [1    0   1];
            actFunIdx2      = [4    4   0];
            numBlocks2      = 3;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    2   6];
            filter3         = [64   64  64];
            kernelSize3     = [3    1   3];
            padding3        = [1    0   1];
            paddingType3    = [1    0   1];
            stride3         = [1    0   1];
            actFunIdx3      = [4    4   0];
            numBlocks3      = 3;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 8;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [64   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = D20KerasWithoutNorm(imgSize, ny)
            % Input
            layer0          = [2                2];
            filter0         = [imgSize(3)       16];
            kernelSize0     = [3                3];
            padding0        = [1                1];
            paddingType0    = [1                1];
            stride0         = [1                1];
            nodes0          = [prod(imgSize)    0];
            actFunIdx0      = [0                4];
            connect0        = [0                0];
            xsc0            = [0                1];
            % Stage 1
            layer1          = [2    2];
            filter1         = [16   16];
            kernelSize1     = [3    3];
            padding1        = [1    1];
            paddingType1    = [1    1];
            stride1         = [1    1];
            actFunIdx1      = [4    4];
            numBlocks1      = 3;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 3;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2   2];
            filter2         = [32  32];
            kernelSize2     = [3   3];
            padding2        = [1   1];
            paddingType2    = [1   1];
            stride2         = [1   1];
            actFunIdx2      = [4   4];
            numBlocks2      = 3;
            tpadding2       = 1;
            tpaddingType2   = 2;
            tstride2        = 2;
            tkernel2        = 3;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    2];
            filter3         = [64   64];
            kernelSize3     = [3    3];
            padding3        = [1    1];
            paddingType3    = [1    1];
            stride3         = [1    1];
            actFunIdx3      = [4    4];
            numBlocks3      = 2;
            tpadding3       = 1;
            tpaddingType3   = 1;
            tstride3        = 1;
            tkernel3        = 3;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Last shortcut
            layer4          = [2    2];
            filter4         = [64   64];
            kernelSize4     = [3    8];
            padding4        = [1    0];
            paddingType4    = [1    2];
            stride4         = [1    1];
            nodes4          = [0    0];
            actFunIdx4      = [4    4];
            connect4        = [0    1];
            xsc4            = [0    1];
            % Output
            layerL          = [4    1];
            filterL         = [64   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layer4          layerL];
            filter          = [filter0        filter1         filter2       filter3         filter4         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSize4     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        padding4        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingType4    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         stride4         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodes4          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdx4      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connect4        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xsc4            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, xsc] = bottleneck(imgSize, ny)
            % Input + 1 CONV
            layer0          = [2                2       6];
            filter0         = [imgSize(3)       16      16];
            kernelSize0     = [3                1       3];
            padding0        = [1                0       1];
            paddingType0    = [1                0       1];
            stride0         = [1                0       1];
            nodes0          = [prod(imgSize)    0       0];
            actFunIdx0      = [0                0       4];
            connect0        = [0                0       0];
            xsc0            = [0                0       1];
            % Stage 1
            layer1          = [2    6   2   6   2   6];
            filter1         = [64   64  64  64  64  64]/4;
            kernelSize1     = [1    1   1   3   1   1];
            padding1        = [0    0   0   1   0   0];
            paddingType1    = [0    0   0   1   0   0];
            stride1         = [0    1   0   1   0   1];
            actFunIdx1      = [4    0   4   0   4   0];
            numBlocks1      = 1;
            tpadding1       = 1;
            tpaddingType1   = 2;
            tstride1        = 2;
            tkernel1        = 1;
            [layer1, filter1, kernelSize1, padding1, paddingType1, stride1, nodes1, actFunIdx1, connect1, xsc1] = resnet.block(layer1, filter1, kernelSize1, padding1, paddingType1, stride1, actFunIdx1,  numBlocks1, tkernel1, tpadding1, tpaddingType1, tstride1);
            % Stage 2
            layer2          = [2    6   2   6   2   6];
            filter2         = [128  128 128 128 128 128]/4;
            kernelSize2     = [1    1   1   3   1   1];
            padding2        = [0    0   0   1   0   0];
            paddingType2    = [0    0   0   1   0   0];
            stride2         = [0    1   0   1   0   1];
            actFunIdx2      = [4    0   4   0   4   0];
            numBlocks2      = 1;
            tpadding2       = 0;
            tpaddingType2   = 0;
            tstride2        = 2;
            tkernel2        = 1;
            [layer2, filter2, kernelSize2, padding2, paddingType2, stride2, nodes2, actFunIdx2, connect2, xsc2] = resnet.block(layer2, filter2, kernelSize2, padding2, paddingType2, stride2, actFunIdx2, numBlocks2, tkernel2, tpadding2, tpaddingType2, tstride2);
            % Satge 3
            layer3          = [2    6   2   6   2   6];
            filter3         = [256  256 256 256 256 256]/4;
            kernelSize3     = [1    1   1   3   1   1];
            padding3        = [0    0   0   1   0   0];
            paddingType3    = [0    0   0   1   0   0];
            stride3         = [0    1   0   1   0   0];
            actFunIdx3      = [4    0   4   0   4   0];
            numBlocks3      = 1;
            tpadding3       = 0;
            tpaddingType3   = 2;
            tstride3        = 1;
            tkernel3        = 9;
            [layer3, filter3, kernelSize3, padding3, paddingType3, stride3, nodes3, actFunIdx3, connect3, xsc3] = resnet.block(layer3, filter3, kernelSize3, padding3, paddingType3, stride3, actFunIdx3, numBlocks3, tkernel3, tpadding3, tpaddingType3, tstride3);
            % Output
            layerL          = [4    1];
            filterL         = [256   1];
            kernelSizeL     = [1    1];
            paddingL        = [0    0];
            paddingTypeL    = [0    0];
            strideL         = [0    0];
            nodesL          = [0    ny];
            actFunIdxL      = [0    0];
            connectL        = [0    0];
            xscL            = [0    0];
            % Stacking
            layer           = [layer0         layer1          layer2        layer3          layerL];
            filter          = [filter0        filter1         filter2       filter3         filterL];
            kernelSize      = [kernelSize0    kernelSize1     kernelSize2   kernelSize3     kernelSizeL];
            padding         = [padding0       padding1        padding2      padding3        paddingL];
            paddingType     = [paddingType0   paddingType1    paddingType2  paddingType3    paddingTypeL];
            stride          = [stride0        stride1         stride2       stride3         strideL];
            nodes           = [nodes0         nodes1          nodes2        nodes3          nodesL];
            actFunIdx       = [actFunIdx0     actFunIdx1      actFunIdx2    actFunIdx3      actFunIdxL];
            connect         = [connect0       connect1        connect2      connect3        connectL];
            xsc             = [xsc0           xsc1            xsc2          xsc3            xscL];
            xscIdx          = find(xsc);
            xscIdx(end)     = [];
            xsc             = zeros(size(connect));
            xsc(connect==1) = xscIdx;
            M               = [layer;filter;kernelSize;padding;paddingType;stride;nodes;actFunIdx];
            check=1;
        end
        function [layer, filter, kernelSize, padding, paddingType, stride, nodes, actFunIdx, connect, xsc] = block(layer, filter, kernelSize, padding, paddingType, stride, actFunIdx, numBlocks, tkernel, tpadding, tpaddingType, tstride)
            n           = length(layer);
            layer       = repmat(layer, [1, numBlocks]);
            filter      = repmat(filter, [1, numBlocks]);
            kernelSize  = repmat(kernelSize, [1, numBlocks]);
            kernelSize(end) = tkernel;
            padding     = repmat(padding, [1, numBlocks]);
            padding(end)= tpadding;
            paddingType = repmat(paddingType, [1, numBlocks]);
            paddingType(end) = tpaddingType;
            stride      = repmat(stride, [1, numBlocks]);
            stride(end) = tstride;
            nodes       = zeros(size(stride));
            actFunIdx   = repmat(actFunIdx, [1, numBlocks]);
            connect     = zeros(size(stride));
            xsc         = zeros(size(stride));
%             connect(end)= 1;
            connect(n:n:length(connect)) = 1;
%             xsc(end)    = 1;
            xsc(n:n:length(connect))     = 1;
            check=1;
        end
    end
end