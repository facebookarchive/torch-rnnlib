-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- A test for Rnnlib's SequenceTable + RecurrentTable
--   This file tests the forward and backward passes.
--------------------------------------------------------------------------------

local rnnlib = require 'rnnlib'
local mutils = require 'rnnlib.mutils'

local rnnlibtest = torch.TestSuite()
local tester     = torch.Tester()

-- These three functions return variants of ax + a, where
-- a is the previous hidden unit state, and x is the input of the layer.
local f2 = function(hid, input)
    local ax    = nn.CMulTable  ()  {hid, input}
    local axpa  = nn.CAddTable  ()  {ax, hid}
    local axpa2 = nn.AddConstant(-2)(axpa)
    return axpa2, axpa
end
local f1 = function(hid, input)
    local ax    = nn.CMulTable  ()  {hid, input}
    local axpa  = nn.CAddTable  ()  {ax, hid}
    local axpa1 = nn.AddConstant(-1)(axpa)
    return axpa1, axpa
end
local f0 = function(hid, input)
    local ax   = nn.CMulTable(){hid, input}
    local axpa = nn.CAddTable(){ax, hid}
    return axpa, nn.Identity()(axpa)
end

-- This is the input, where each index `i` represents the `ith` dimension's
-- initial values.
local input = {
    [1] = {
        -- state
        [1] = torch.Tensor{4},
        [2] = torch.Tensor{3},
        [3] = torch.Tensor{2},
    },
    [2] = {
        -- input
        torch.Tensor{1},
        torch.Tensor{2},
    }
}

-- The partial computation is accumulated over all dimensions and returned
-- in the output.
local output = {
    [1] = { -- hidden
        -- timesteps
        [1] = {
            -- hidden states along depth
            torch.Tensor{6},  -- 4 * 1  + 4 - 2
            torch.Tensor{26}, -- 3 * 8  + 3 - 1
            torch.Tensor{56}, -- 3 * 27 + 2 - 0
        },
        [2] = {
            torch.Tensor{16},    -- 6  * 2   + 6  - 2
            torch.Tensor{493},   -- 26 * 18  + 26 - 1
            torch.Tensor{27720}, -- 56 * 494 + 56 - 0
        },
    },
    [2] = { -- output
        -- timesteps
        [1] = {
            -- outputs along depth
            torch.Tensor{8},  -- 4 * 1  + 4
            torch.Tensor{27}, -- 3 * 8  + 3
            torch.Tensor{56}, -- 2 * 27 + 2
        },
        [2] = {
            torch.Tensor{18},    -- 6  * 2   + 6
            torch.Tensor{494},   -- 26 * 18  + 26
            torch.Tensor{27720}, -- 56 * 494 + 56
        },
    },
}

function rnnlibtest.axpa()
    local modeloutput = torch.Tensor{56, 27720}

    -- Unfinished example.
    local gradOutput = {
        [1] = { -- hidden
            -- timesteps
            [1] = {
                -- hidden states along depth
                torch.Tensor{1},
                torch.Tensor{1},
                torch.Tensor{1},
            },
            [2] = {
                torch.Tensor{1},
                torch.Tensor{1},
                torch.Tensor{1},
            },
        },
        [2] = { -- output
            -- timesteps
            [1] = {
                -- outputs along depth
                torch.Tensor{1},
                torch.Tensor{1},
                torch.Tensor{1},
            },
            [2] = {
                torch.Tensor{1},
                torch.Tensor{1},
                torch.Tensor{1},
            },
        },
    }

    -- For the first example, we construct a model such that its outer loop is
    -- time. This means we must first build the network over depth.

    local depth = nn.SequenceTable{dim=1}

    depth:add(rnnlib.cell.gModule(f2))
    depth:add(rnnlib.cell.gModule(f1))
    depth:add(rnnlib.cell.gModule(f0))

    local m1 = nn.RecurrentTable{module=depth, dim=2}

    -- Indexing into the output of the model must be done carefully.
    -- `mutils.outmodule` will map the module given in the argument
    -- across the outermost dimension.
    --
    -- For this example, we are only interested in the output of the topmost
    -- layer of the network at each timestep.
    local model1 = nn.Sequential()
        :add(mutils.outmodule(m1, nn.SelectTable(-1)))
        :add(nn.SelectTable(2))
        :add(nn.JoinTable(1))
    model1:forward(input)

    tester:eq(m1.output, output)
    tester:eq(model1.output, modeloutput)

    m1:backward(input, gradOutput)

    -- Next we have the outer loop be over depth. We build the network layer-wise.

    local layers = {
        nn.RecurrentTable{module=rnnlib.cell.gModule(f2), dim=2},
        nn.RecurrentTable{module=rnnlib.cell.gModule(f1), dim=2},
        nn.RecurrentTable{module=rnnlib.cell.gModule(f0), dim=2},
    }

    local m2 = nn.SequenceTable{modules=layers, dim=1}

    local model2 = nn.Sequential()
        :add(m2)
        :add(nn.SelectTable(2))
        :add(nn.SelectTable(-1))
        :add(nn.JoinTable(1))

    model2:forward(input)

    tester:eq(model2.output, model1.output)
end

function rnnlibtest.inmodule()
    local input1 = mutils.deepcopy(input)

    -- Scale all tensors in a table by 2.
    local function t2(x)
        if torch.isTensor(x) then
            x:mul(2)
        else
            for k, v in pairs(x) do
                t2(v)
            end
        end
    end
    t2(input1[2])

    -- Create a normal model without an inmodule.
    local layers1 = {
        nn.RecurrentTable{module = rnnlib.cell.gModule(f2), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f1), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f0), dim = 2},
    }

    local m1 = nn.SequenceTable{modules = layers1, dim = 1}

    local model1 = nn.Sequential()
        :add(m1)
        :add(nn.SelectTable(2))
        :add(nn.SelectTable(-1))
        :add(nn.JoinTable(1))

    local input2 = mutils.deepcopy(input)

    -- Rather than scaling the input manually, use `mutils.inmodule`.
    local layers2 = {
        nn.RecurrentTable{module = rnnlib.cell.gModule(f2), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f1), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f0), dim = 2},
    }

    local m2 = nn.SequenceTable{modules = layers2, dim = 1}
    local model2 = nn.Sequential()
        :add(mutils.inmodule(m2, nn.MulConstant(2)))
        :add(nn.SelectTable(2))
        :add(nn.SelectTable(-1))
        :add(nn.JoinTable(1))

    local input3 = mutils.deepcopy(input)

    -- Concatenate the inputs so that the input is of dimension seqlen x inputsize.
    for k, v in pairs(input3[2]) do
        input3[2][k] = v:view(1, 1, 1)
    end
    input3[2] = torch.cat(input3[2], 1)

    -- Rather than scaling the input manually, use `mutils.batchedinmodule`.
    local layers3 = {
        nn.RecurrentTable{module = rnnlib.cell.gModule(f2), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f1), dim = 2},
        nn.RecurrentTable{module = rnnlib.cell.gModule(f0), dim = 2},
    }

    local m3 = nn.SequenceTable{modules = layers3, dim = 1}
    local model3 = nn.Sequential()
        :add(mutils.batchedinmodule(m3, nn.MulConstant(2)))
        :add(nn.SelectTable(2))
        :add(nn.SelectTable(-1))
        :add(nn.JoinTable(1))

    model1:forward(input1)
    model2:forward(input2)
    model3:forward(input3)

    tester:eq(model1.output, model2.output)
    tester:eq(model1.output, model3.output)
end

tester:add(rnnlibtest)
tester:run()
