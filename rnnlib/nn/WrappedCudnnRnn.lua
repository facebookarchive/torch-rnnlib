-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- nn.WrappedCudnnRnn
--   Wraps a sequence of Cudnn.RNNs so that the API is the same as Rnnlib's.
--------------------------------------------------------------------------------

require 'nn'
assert(pcall(require, 'cudnn'), 'You must have the torch Cudnn bindings')

local argcheck = require 'argcheck'
local cutils   = require 'rnnlib.cudnnutils'

local WrappedCudnnRnn, parent = torch.class('nn.WrappedCudnnRnn', 'nn.Sequential')

-- | Wrap a list of cudnn modules that form a chain.
-- The keys of a table are an alias for the elements inside of it,
-- while the types after the colon represent the dimensions.
-- If a dimension is wrapped in a table, then that dimension is a
-- 'table dimension' and it represents the number of elements in that table.
-- The input to the wrapped module will be of type:
--     {
--         { hiddens : {nmodules} x nlayer_i x bsz x hidden },
--         { input   : {bptt}     x bsz      x emsize },
--     }.
-- Note that this is different from Rnnlib's input type, which is
--     {
--         { hiddens : {nlayer} x bsz x hidden },
--         { input   : {bptt}   x bsz x emsize },
--     }.

-- Since only the last layer's output will be returned, the sequence
-- output will be a singleton table with a tensor of dimension bptt x hidden
-- (represented in our notation by {1} x bptt x hidden).
-- The output of the wrapped module will be of type:
--     {
--         { newHiddens : {nmodules} x nlayer_i x bsz x hidden },
--         { output     : {1}        x {bptt}   x bsz x hidden },
--     }.
-- WARNING: This is different from Rnnlib's API, which returns an output of type
--     {
--         { newHiddens : {nlayer} x {bptt} x bsz x hidden },
--         { output     : {nlayer} x {bptt} x bsz x hidden },
--     }.
-- modules : A table of objects that are subtypes of cudnn.RNN.
WrappedCudnnRnn.__init = argcheck{
    { name = 'self'       , type = 'nn.WrappedCudnnRnn' },
    { name = 'modules'    , type = 'table'              },
    { name = "savehidden" , type = "boolean"            },
    call = function(self, modules, savehidden)
        assert(#modules >= 1, "There must be at least one cudnn module.")
        parent.__init(self)
        local nmodules = #modules

        -- This joins the across the table dimension of the input
        -- (which has dimension {bptt} x bsz x emsize)
        -- to create a tensor of dimension bptt x bsz x emsize.
        self
            :add(nn.MapTable(nn.View(1, -1, modules[1].inputSize)))
            :add(nn.JoinTable(1))
        -- Stack the cudnn modules on top of each other.
        for i = 1, nmodules do
            self:add(modules[i])
        end
        -- Split the output from bptt x bsz x hidden to {bptt} x bsz x hidden.
        self:add(nn.SplitTable(1))

        self.rnns     = modules
        self.nmodules = nmodules

        self.saveHidden = savehidden
    end
}

-- | Convert an existing nn.SequenceTable model with a single type of cell
-- to Cudnn. The `cellstring` must be RNN{Tanh, ReLU}, GRU or LSTM.
WrappedCudnnRnn.__init = argcheck{
    { name = 'self'       , type = 'nn.WrappedCudnnRnn' },
    { name = 'model'      , type = 'nn.SequenceTable'   },
    { name = 'cellstring' , type = 'string'             },
    { name = 'hids'       , type = 'table'              },
    { name = "savehidden" , type = "boolean"            },
    overload = WrappedCudnnRnn.__init,
    call = function(self, model, cellstring, hids, savehidden)
        local oldparams = model:parameters()
        local nlayers   = #hids
        -- If the network is homogeneous, i.e. that all hidden layers have the
        -- same dimensionality.
        local allsame   = true
        for i = 1, nlayers - 1 do
            allsame = hids[i] == hids[i+1] and allsame
        end

        local rnn
        if allsame then
            rnn = cudnn[cellstring](hids[0], hids[1], nlayers)
            for l = 1, nlayers do
                -- 2*l-1 = input  params for layer i
                -- 2*l   = hidden params for layer i
                cutils.copyParams(
                    rnn, cutils.offsets[cellstring],
                    oldparams[2*l-1], oldparams[2*l],
                    hids[l], l)
            end
            -- Create a singleton list, since wrapcudnnmodules
            -- only takes a table of cudnn modules.
            rnn = self.__init(self, {rnn}, savehidden)
        else
            -- The layers must be constructed individually because the Cudnn API
            -- expects RNNs to all have the same dimension.
            rnn = {}
            for l = 1, nlayers do
                local layer = cudnn[cellstring](hids[l-1], hids[l], 1)
                cutils.copyParams(
                    layer, cutils.offsets[cellstring],
                    oldparams[2*l-1], oldparams[2*l],
                    hids[l], 1)
                rnn[l] = layer
            end
            rnn = self.__init(self, rnn, savehidden)
        end
        return rnn
    end
}

-- | Initialize the hidden state to zero.
WrappedCudnnRnn.initializeHidden = function(self, batchsize)
    local nmodules = self.nmodules
    local modules  = self.rnns
    local hiddenbuffer = self.hiddenbuffer or {}
    for i = 1, nmodules do
        local dim = {
            modules[i].numLayers,
            batchsize,
            modules[i].hiddenSize,
        }
        if modules[i].mode:find('LSTM') then
            hiddenbuffer[i] = hiddenbuffer[i]
                or {
                    torch.CudaTensor(),
                    torch.CudaTensor(),
                }
            hiddenbuffer[i][1]:resize(table.unpack(dim)):fill(0)
            hiddenbuffer[i][2]:resize(table.unpack(dim)):fill(0)
        else
            hiddenbuffer[i] = hiddenbuffer[i] or torch.CudaTensor()
            hiddenbuffer[i]   :resize(table.unpack(dim)):fill(0)
        end
    end
    self.hiddenbuffer = hiddenbuffer
    return hiddenbuffer
end

-- | Save hidden state.
WrappedCudnnRnn.saveLastHidden = function(self)
    local nmodules = self.nmodules
    local modules  = self.rnns
    local bsz = modules[1].hiddenOutput:size(2)
    local hiddenbuffer = self.hiddenbuffer
        or self:initializeHidden(bsz)
    for i = 1, nmodules do
        if modules[i].mode:find('LSTM') then
            hiddenbuffer[i][1]
                :resizeAs(modules[i].cellOutput)
                :copy(modules[i].cellOutput)
            hiddenbuffer[i][2]
                :resizeAs(modules[i].hiddenOutput)
                :copy(modules[i].hiddenOutput)
        else
            hiddenbuffer[i]
                :resizeAs(modules[i].hiddenOutput)
                :copy(modules[i].hiddenOutput)
        end
    end
    return hiddenbuffer
end

-- | Get the last hidden state.
WrappedCudnnRnn.getLastHidden = function(self)
    return self.hiddenbuffer
end

-- | Overload updateOutput to set the hidden states.
WrappedCudnnRnn.updateOutput = function(self, input)
    local nmodules = self.nmodules
    local modules  = self.rnns

    local hidinput = input[1]
    local seqinput = input[2]
    for i = 1, nmodules do
        if modules[i].mode:find('LSTM') then
            modules[i].cellInput   = hidinput[i][1]
            modules[i].hiddenInput = hidinput[i][2]
        else
            modules[i].hiddenInput = hidinput[i]
        end
    end

    local seqoutput = parent.updateOutput(self, seqinput)
    local hidoutput = self.hidoutput or {}
    for i = 1, nmodules do
        hidoutput[i] = modules[i].mode:find('LSTM')
            and {
                modules[i].cellOutput, modules[i].hiddenOutput,
            }
            or modules[i].hiddenOutput
    end
    for i = nmodules+1, #hidoutput do
        hidoutput[i] = nil
    end
    self.hidoutput = hidoutput
    if not self.train and self.saveHidden then
        self:saveLastHidden()
    end
    -- `seqoutput` is wrapped in another table because the Cudnn API only
    -- returns the final layer's output. The Rnnlib API requires each layer
    -- to output a table, so we treat the Cudnn API as a single layer.
    -- This could be circumvented using SequenceTable and the Cudnn API to
    -- make each layer individually, but is a non-negligible amount slower
    -- due to the increased amount of copies.
    self.output = { hidoutput, { seqoutput } }
    return self.output
end

WrappedCudnnRnn.updateGradInput = function(self, input, gradOutput)
    local nmodules      = self.nmodules
    local modules       = self.rnns
    local seqinput      = input[2]
    local hidgradoutput = gradOutput[1]
    -- Recall the wrapping from above, where the sequence output is wrapped
    -- in another table to conform with the Rnnlib standard of having there
    -- be one table of outputs for each hidden layer. We treat a stack of
    -- Cudnn modules as a single hidden layer since only the final output is
    -- returned.
    local seqgradoutput = gradOutput[2][1]
    for i = 1, nmodules do
        if modules[i].mode:find('LSTM') then
            modules[i].gradCellOutput   = hidgradoutput[i][1]
            modules[i].gradHiddenOutput = hidgradoutput[i][2]
        else
            modules[i].gradHiddenOutput = hidgradoutput[i]
        end
    end

    local seqgradinput = parent.updateGradInput(self, seqinput, seqgradoutput)
    local hidgradinput = {}
    for i = 1, nmodules do
        hidgradinput[i] = modules[i].mode:find('LSTM')
            and {
                modules[i].gradCellInput, modules[i].gradHiddenInput,
            }
            or modules[i].gradHiddenInput
    end
    self.gradInput = { hidgradinput, seqgradinput }
    return self.gradInput
end

WrappedCudnnRnn.accGradParameters = function(self, input, gradOutput, scale)
    local nmodules      = self.nmodules
    local modules       = self.rnns
    local seqinput      = input[2]
    local hidgradoutput = gradOutput[1]
    local seqgradoutput = gradOutput[2][1]
    for i = 1, nmodules do
        if modules[i].mode:find('LSTM') then
            modules[i].gradCellOutput   = hidgradoutput[i][1]
            modules[i].gradHiddenOutput = hidgradoutput[i][2]
        else
            modules[i].gradHiddenOutput = hidgradoutput[i]
        end
    end
    parent.accGradParameters(self, seqinput, seqgradoutput, scale)
    -- Zero out gradBias to conform with Rnnlib standard which does
    -- not use biases in linear projections.
    for i = 1, #modules do
        cutils.zeroField(modules[i], 'gradBias')
    end
    if self.train and self.saveHidden then
        self:saveLastHidden()
    end
end

-- | The backward must be overloaded because nn.Sequential's backward does not
-- actually call updateGradInput or accGradParameters.
WrappedCudnnRnn.backward = function(self, input, gradOutput, scale)
    local gradInput = self:updateGradInput(input, gradOutput)
    self:accGradParameters(input, gradOutput, scale)
    return gradInput
end
