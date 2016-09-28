-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Utilities for using Cudnn with Rnnlib
--------------------------------------------------------------------------------

require 'nn'
assert(pcall(require, 'cudnn'), 'You must have the torch Cudnn bindings.')

local ffi = require 'ffi'

local utils = {}

-- The number of linear projections in each model.
local numProjs = {
    -- Shorthands
    RNN     = 2,
    RNNTanh = 2,
    RNNReLU = 2,
    GRU     = 6,
    LSTM    = 8,

    -- Cudnn modes
    CUDNN_RNN_TANH = 2,
    CUDNN_RNN_RELU = 2,
    CUDNN_GRU      = 6,
    CUDNN_LSTM     = 8,
}
utils.numProjs = numProjs

-- See cudnn r5 library user guide.
local offsets = {
    RNN     = { input = {0}, hidden = {1}, },
    RNNTanh = { input = {0}, hidden = {1}, },
    RNNReLU = { input = {0}, hidden = {1}, },

    GRU     = {
        input  = { r = 0, u = 1, c = 2, },
        hidden = { r = 3, u = 4, c = 5, },
    },

    LSTM    = {
        input  = { i = 0, f = 1, c = 2, o = 3, },
        hidden = { i = 4, f = 5, c = 6, o = 7, },
    },
}
utils.offsets = offsets

local function firstToUpper(str)
    return str:gsub("^%l", string.upper)
end

-- | Get the parameters for a linear projection.
-- rnn     : The Cudnn module.
-- layer   : The layer number (0 indexed).
-- layerId : The projection in the layer (0 indexed).
local function getParams(rnn, layer, layerId)
    if not rnn.wDesc then
        rnn:resetWeightDescriptor()
    end
    local fns = {
        weight = 'cudnnGetRNNLinLayerMatrixParams',
        bias   = 'cudnnGetRNNLinLayerBiasParams',
    }
    local params = {}
    for key, fn in pairs(fns) do
        local desc = rnn:createFilterDescriptors(1)
        local pointer = ffi.new("float*[1]")
        cudnn.errcheck(
            fn,
            cudnn.getHandle(),
            rnn.rnnDesc[0],
            layer,
            rnn.xDescs[0],
            rnn.wDesc[0],
            rnn.weight:data(),
            layerId,
            desc[0],
            ffi.cast("void**", pointer)
        )

        local dataType = ffi.new("cudnnDataType_t[1]")
        local format   = ffi.new("cudnnTensorFormat_t[1]")
        local nbDims   = torch.IntTensor(1)

        local minDim = 3
        local filterDimA = torch.ones(minDim):int()
        cudnn.errcheck(
            'cudnnGetFilterNdDescriptor',
            desc[0],
            minDim,
            dataType,
            format,
            nbDims:data(),
            filterDimA:data()
        )

        local offset = pointer[0] - rnn.weight:data()
        params[key] = torch.CudaTensor(
            rnn.weight:storage(), offset + 1, filterDimA:prod())
        params["grad" .. firstToUpper(key)] = torch.CudaTensor(
            rnn.gradWeight:storage(), offset + 1, filterDimA:prod())
    end
    return params
end
utils.getParams = getParams

-- | Copy parameters from an rnnlib model to a cudnn one.
-- If the rnnlib network has different hidden dimensionality on each layer
-- then the parameters must be copied over to cudnn one layer at a time.
-- rnn     : The cudnn module.
-- offset  : The projection offsets (see Cudnn r5 library user guide).
-- oldin   : The input parameters from the rnnlib module.
-- oldhid  : The hidden parameters from the rnnlib module.
-- hidSize : The hidden size of the layer (or network, if homogeneous).
-- nlayer  : The number layers in the network.
function utils.copyParams(rnn, offset, oldin, oldhid, hidSize, nlayer)
    local inps = oldin :split(hidSize, 1)
    local hids = oldhid:split(hidSize, 1)
    local ngates = #inps
    for dir, t in pairs(offset) do
        local old = dir == "input" and inps or hids
        for gate, id in pairs(t) do
            local params = getParams(rnn, nlayer-1, id)
            params.weight:copy(old[id % ngates + 1]:view(-1))
            params.bias:fill(0)
        end
    end
end

-- | A utility function to zero out a module's tensors.
-- rnn   : The cudnn module.
-- field : The parameter to be zeroed out [weight, bias, gradWeight, gradBias].
function utils.zeroField(rnn, field)
    for i = 0, rnn.numLayers-1 do
        for j = 0, numProjs[rnn.mode] - 1 do
            local ps = getParams(rnn, i, j)
            ps[field]:fill(0)
        end
    end
    rnn:resetWeightDescriptor()
end

return utils
