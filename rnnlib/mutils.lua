-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local doc = require 'argcheck.doc'

doc [[
These are a collection of model utility functions for constructing and
manipulating SequenceTables as well as the input and output of said module.
]]

local utils = {}

--------------------------------------------------------------------------------
-- Utility functions
--------------------------------------------------------------------------------

-- | Only transpose the first two dimensions.
-- This function is identical to zip.
local transpose = function(x)
    local y = {}
    for i = 1, #x do
        for j = 1, #x[1] do
            y[j] = y[j] or {}
            y[j][i] = x[i][j]
        end
    end
    return y
end
utils.transpose = transpose

-- | Perform a deep copy where both tensors and tables are structurally
-- replicated.
local function _deepcopy(x)
    if torch.isTensor(x) then
        return x:clone()
    else
        local ntbl = {}
        for k, v in pairs(x) do
            ntbl[k] = _deepcopy(v)
        end
        return ntbl
    end
end

utils.deepcopy = _deepcopy

-- | Given two tables of tensors (arbitrarily deep) with the same structure,
-- resize and copy the second argument's tensors `y` into the first `x`.
local function _resizeCopy(x, y)
    if torch.isTensor(x) then
        x:typeAs(y):resizeAs(y):copy(y)
    else
        for k, _ in pairs(y) do
            _resizeCopy(x[k], y[k])
        end
    end
end

utils.resizeCopy = _resizeCopy

--------------------------------------------------------------------------------
-- Initialization helpers
--------------------------------------------------------------------------------

-- | The default initialization function.
utils.defwinitfun = function(model, range)
    range = range or 0.1
    local w = model:parameters()
    for i, v in pairs(w) do
        w[i] = v:uniform(-range, range)
    end
end

-- | Recursive init on a per-module basis.
-- initT : Map ModuleType (Weight -> Weight)
utils.scalewinitfun = function(module, initT)
    -- initT should look something like:
    -- initT = {
    --     ['nn.Linear'] = function(weight)
    --         weight:uniform(-0.1, 0.1)
    --     end,
    --     ['nn.LookupTable'] = function(weight)
    --         ...
    --     end,
    --     ...
    -- }
    local winit = function(module)
        local winitfn = initT[torch.typename(module)]
        if winitfn ~= nil then
            local p = module:parameters()
            for k, v in pairs(p) do
                winitfn(v)
            end
        end
    end
    module:apply(winit)
end

--------------------------------------------------------------------------------
-- Wrapping functions for recurrent layers
--------------------------------------------------------------------------------

local SeqType = "nn.Sequential"
local SetType = "nn.SequenceTable"
local RnnType = "nn.RecurrentTable"

-- | Any elements f(x) == true.
local any = function(t, f)
    for _, v in pairs(t) do
        if f(v) == true then return true end
    end
    return false
end

-- | If the top level model is cudnn or any of its children.
-- Will fix later.
local function isCudnn(model)
    return torch.isTypeOf(model, "cudnn.*")
        or any(model.modules, function(x)
             return torch.isTypeOf(x, "cudnn.*")
        end)
end

-- | Asserts that an RNN is wrapped correctly or wraps an unwrapped one.
local assertWrapped = function(model)
    if
        torch.isTypeOf(model, RnnType)
        or torch.isTypeOf(model, SetType)
        or isCudnn(model)
    then
        -- need to wrap with Sequential
        model = nn.Sequential():add(model)
    elseif torch.isTypeOf(model, SeqType) then
        -- already wrapped with Sequential
        assert(
            #model.modules <= 3,
            string.format(
                "You have %d modules in your RNN, you should have at most inmodule + RNN + outmodule: %s ",
                #model.modules, tostring(model)
            )
        )
        assert(
            torch.isTypeOf(model:get(1), RnnType)
                or torch.isTypeOf(model:get(2), RnnType)
                or torch.isTypeOf(model:get(1), SetType)
                or torch.isTypeOf(model:get(2), SetType)
                or any(model.modules, isCudnn),
            string.format(
                "Your rnn should be in either module 1 or module 2. Instead, your model is %s",
                tostring(model)
            )
        )
    else
        error(string.format(
            "The type of your model is %s, while Sequential or Recurrent is expected.",
            torch.typename(model)
        ))
    end
    return model
end

-- | Pipes the input through a module.
-- The model must pass the assertWrapped tests.
-- The inmodule is applied to the part of the input that does not belong to the
-- hidden state, which is assumed to be in index 2 of the input.
utils.inmodule = function(model, inmodule)
    model = assertWrapped(model)
    local inm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.MapTable(inmodule))
    table.insert(model.modules, 1, inm)
    return model
end

-- | Pipes a batch through a module and splits along the first dimension.
-- The inmodule is applied to an input tensor of seqlen x nbatch x hidden.
utils.batchedinmodule = function(model, inmodule)
    model = assertWrapped(model)
    local inm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.Sequential()
             :add(inmodule)
             :add(nn.SplitTable(1))
        )
    table.insert(model.modules, 1, inm)
    return model
end

-- | Applies a module to the output table along the outermost dimension.
utils.outmodule = function(model, outmodule)
    model = assertWrapped(model)
    local outm = nn.ParallelTable()
        :add(nn.Identity())
        :add(nn.MapTable(outmodule))

    return model:add(outm)
end

--------------------------------------------------------------------------------
-- Utils for RNN Hidden State
--------------------------------------------------------------------------------

-- | Saves the last hidden state of a two dimensional RNN.
-- A two dimensional RNN is one which takes as input a table of the previous
-- hidden state as well as a sequence. The two dimensions are the depth of the
-- RNN (how many layers are stacked) and the length of the input sequence.
utils.makeSaveLastHidden = function(depthfirst)
    depthfirst = depthfirst == nil and false or depthfirst
    return function(self)
        local lasthid = self:getLastHidden()
        if self.hiddenbuffer ~= nil then
            _resizeCopy(self.hiddenbuffer, lasthid)
        else
            self.hiddenbuffer = _deepcopy(lasthid)
        end
    end
end

-- | Gets the last hidden state of a two dimensional RNN.
utils.makeGetLastHidden = function(depthfirst)
    depthfirst = depthfirst == nil and false or depthfirst
    return function(self)
        local hids = self.output[1]
        return depthfirst
            and hids[#hids]
            or  transpose(hids)[#hids[1]]
    end
end

return utils
