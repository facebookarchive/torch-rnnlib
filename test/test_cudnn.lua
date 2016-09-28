-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Testing Script for RNN CUDNN Bindings
--------------------------------------------------------------------------------

require 'math'
require 'cudnn'

local cell   = require 'rnnlib.cell'
local rnnlib = require 'rnnlib'

local cudnntest = torch.TestSuite()
local tester = torch.Tester()

local tol = 1e-6

--------------------------------------------------------------------------------
-- Utility functions
--------------------------------------------------------------------------------

local function reccopy(x)
    if torch.isTensor(x) then
        return x:clone()
    else
        local new = {}
        for k, v in pairs(x) do
            new[k] = reccopy(v)
        end
        return new
    end
end

local function recfill(x, val)
    if torch.isTensor(x) then
        return x:fill(val)
    else
        for k, v in pairs(x) do
            recfill(v, val)
        end
    end
end

function math.atanh(x)
    return (math.log(1+x) - math.log(1-x)) / 2
end

-- | Copy hidden state from rnnlib2 to a format amenable to cudnn.
local rnnlib2cudnn4hid = function(hid, cellstring)
    local nlayer = #hid
    local allsame = true
    for i = 1, nlayer-1 do
        if cellstring:find("LSTM") then
            allsame = hid[i][1]:size(2) == hid[i+1][1]:size(2) and allsame
        else
            allsame = hid[i]:size(2) == hid[i+1]:size(2) and allsame
        end
    end

    local cellInput, hiddenInput
    if allsame then
        -- For homogeneous networks.
        if cellstring:find("LSTM") then
            cellInput   = torch.CudaTensor(
                nlayer, hid[1][1]:size(1), hid[1][1]:size(2))
            hiddenInput = torch.CudaTensor(
                nlayer, hid[1][2]:size(1), hid[1][2]:size(2))
        else
            hiddenInput = torch.CudaTensor(
                nlayer, hid[1]:size(1), hid[1]:size(2))
        end

        for layer = 1, nlayer do
            if cellstring:find("LSTM") then
                cellInput  [layer]:copy(
                    hid[layer][1]:view(
                        1, hid[layer][1]:size(1), hid[layer][1]:size(2)))
                hiddenInput[layer]:copy(
                    hid[layer][2]:view(
                        1, hid[layer][2]:size(1), hid[layer][2]:size(2)))
            else
                hiddenInput[layer]:copy(
                    hid[layer]:view(
                        1, hid[layer]:size(1), hid[layer]:size(2)))
            end
        end

        return cellstring:find('LSTM')
            and { { cellInput, hiddenInput } }
            or  { hiddenInput }
    else
        -- For heterogeneous networks.
        -- recursive clone, then change view.
        local chid = {}
        for layer = 1, nlayer do
            if cellstring:find("LSTM") then
                chid[layer] = {}
                chid[layer][1] = hid[layer][1]:clone():view(
                    1, hid[layer][1]:size(1), hid[layer][1]:size(2))
                chid[layer][2] = hid[layer][2]:clone():view(
                    1, hid[layer][2]:size(1), hid[layer][2]:size(2))
            else
                chid[layer] = hid[layer]:clone():view(
                    1, hid[layer]:size(1), hid[layer]:size(2))
            end
        end
        return chid
    end
end

-- Wrap a recurrent model to only output the last layer.
local function wrapmodel(model)
    return nn.Sequential()
        :add(model)
        :add(nn.ParallelTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(nn.SelectTable(-1))
                :add(nn.JoinTable(1))
            )
        )
end

local function assertOutputEq(
        rnnliboutput, cudnnoutput,
        time, cellstring, allsame, i)
    local rhidoutput = rnnliboutput[1]
    local chidoutput = cudnnoutput [1]
    local rseqoutput = rnnliboutput[2]
    local cseqoutput = cudnnoutput [2]
    local nlayer = #rhidoutput
    if cellstring:find("LSTM") then
        for depth = 1, nlayer do
            tester:eq(
                rhidoutput[depth][time][1],
                allsame
                    and chidoutput[1][1][depth]
                    or  chidoutput[depth][1]:squeeze(),
                tol,
                string.format("%s cell at depth %i in batch %i",
                              cellstring, depth, i)
            )
            tester:eq(
                rhidoutput[depth][time][2],
                allsame
                    and chidoutput[1][2][depth]
                    or  chidoutput[depth][2]:squeeze(),
                tol,
                string.format("%s hidden at depth %i in batch %i",
                              cellstring, depth, i)
            )
        end
    else
        for depth = 1, nlayer do
            tester:eq(
                rhidoutput[depth][time],
                allsame
                    and chidoutput[1][depth]
                    or  chidoutput[depth][1]:squeeze(),
                tol,
                string.format("%s hidden at depth %i in batch %i",
                              cellstring, depth, i)
            )
        end
    end
    tester:eq(
        rseqoutput,
        cseqoutput,
        tol,
        string.format("%s output in batch %i", cellstring, i)
    )
end

local function assertGradientEq(
        rnnlibm, cudnnm, rg, cg, cellstring, bsz, allsame, i)
    tester:eq(
        nn.MapTable(nn.View(1, bsz, -1):cuda())
            :forward(rnnlibm.gradInput[2]),
        cudnnm.gradInput[2],
        tol,
        string.format("%s gradInput in batch %i", cellstring, i)
    )

    local rgw = torch.cat(
        nn.MapTable(nn.View(-1):cuda()):forward(rg), 1)
    -- If not allsame, then the parameters are separated by layer.
    local cgw
    if allsame then
        cgw = cg[1]:narrow(1, 1, rgw:size(1))
    else
        cgw = torch.cat({
            cg[1]:narrow(1, 1, rg[1]:nElement() + rg[2]:nElement()),
            cg[2]:narrow(1, 1, rg[3]:nElement() + rg[4]:nElement()),
        }, 1)
    end

    tester:assert(rgw:eq(math.huge):sum() == 0,
                  "Infs in rnnlib gradWeight")
    tester:assert(cgw:eq(math.huge):sum() == 0,
                  "Infs in cudnn gradWeight")
    -- Assert gradients wrt weights are the same.
    tester:eq(
        rgw, cgw, tol,
        string.format("%s gradWeight in batch %i", cellstring, i))
end

local function createGradOutput(
        initfun, rnnliboutput, cudnnoutput, cudnnrnn,
        time, cellstring, allsame)
    local chidoutput = cudnnoutput [1]
    local cseqoutput = cudnnoutput [2]
    local gradOutput = initfun(cseqoutput:clone())
    local gradHidOut = {}
    if cellstring:find("LSTM") then
        for i = 1, #cudnnrnn.rnns do
            gradHidOut[i] = {}
            gradHidOut[i][1] = initfun(chidoutput[i][1]:clone())
            gradHidOut[i][2] = initfun(chidoutput[i][2]:clone())
        end
    else
        for i = 1, #cudnnrnn.rnns do
            gradHidOut[i] = initfun(chidoutput[i]:clone())
        end
    end
    -- Copy errors to rnnlib.
    local rgradOutput = reccopy(rnnliboutput)
    rgradOutput[2]:copy(gradOutput)
    recfill(rgradOutput[1], 0)

    local nlayer = #rnnliboutput[1]
    if allsame then
        if cellstring:find("LSTM") then
            for i = 1, nlayer do
                rgradOutput[1][i][time][1]:copy(gradHidOut[1][1][i])
                rgradOutput[1][i][time][2]:copy(gradHidOut[1][2][i])
            end
        else
            for i = 1, nlayer do
                rgradOutput[1][i][time]:copy(gradHidOut[1][i])
            end
        end
    else
        if cellstring:find("LSTM") then
            for i = 1, nlayer do
                rgradOutput[1][i][time][1]:copy(gradHidOut[i][1][1])
                rgradOutput[1][i][time][2]:copy(gradHidOut[i][2][1])
            end
        else
            for i = 1, nlayer do
                rgradOutput[1][i][time]:copy(gradHidOut[i][1])
            end
        end
    end
    return rgradOutput, { gradHidOut, gradOutput }
end

--------------------------------------------------------------------------------
-- Test cases
--------------------------------------------------------------------------------

-- | This function creates a test for a certain cell.
local function makeTest(
        rnnlibrnn, hids, time, bsz, nbatch,
        cellstring, tieHidden, updateParams, usebackward, initfun)
    local allsame = hids[2] == nil or hids[1] == hids[2]
    return function()
        local ins = hids[0]
        -- Construct cudnn model.
        -- Cudnnrnn will either be a cudnn module or a nn.SequenceTable
        -- composed of cudnn modules to allow for layers with different hidden
        -- dimensions.
        local cudnnrnn = nn.WrappedCudnnRnn(rnnlibrnn, cellstring, hids, true)

        -- Wrap models.
        local rnnlibm = wrapmodel(rnnlibrnn)
        local cudnnm  = wrapmodel(cudnnrnn)

        rnnlibm:cuda():training()
        cudnnm :cuda():training()

        -- Initialize hidden states.
        rnnlibrnn:initializeHidden(bsz)
        cudnnrnn :initializeHidden(bsz)

        for i = 1, nbatch do
            -- Initialize input.
            local cudnninput  = initfun(torch.CudaTensor(time, bsz, ins))
            -- Split the input along sequence length.
            local rnnlibinput = cudnninput:split(1)
            for i = 1, #rnnlibinput do
                rnnlibinput[i] = rnnlibinput[i]:view(bsz, -1)
            end

            -- Prepare hidden states.
            local hid  = rnnlibrnn.hiddenbuffer
            local chid = tieHidden
                and rnnlib2cudnn4hid(hid, cellstring)
                or  cudnnrnn.hiddenbuffer

            -- Perform forward pass.
            local rnnliboutput = rnnlibm:forward{ hid, rnnlibinput }
            local cudnnoutput  = cudnnm :forward{ chid, cudnninput:split(1) }

            -- Assert all outputs are the same.
            assertOutputEq(
                rnnliboutput, cudnnoutput, time, cellstring, allsame, i)

            -- Set up errors.
            local rgradOutput, cgradOutput = createGradOutput(
                initfun, rnnliboutput, cudnnoutput, cudnnrnn,
                time, cellstring, allsame)

            -- Perform backward pass.
            rnnlibm:zeroGradParameters()
            cudnnm :zeroGradParameters()
            if usebackward then
                rnnlibm:backward({ hid, rnnlibinput }, rgradOutput)
                cudnnm :backward(
                    { chid, cudnninput:split(1) },
                    cgradOutput)
            else
                rnnlibm:updateGradInput({ hid, rnnlibinput }, rgradOutput)
                cudnnm :updateGradInput(
                    { chid, cudnninput:split(1) },
                    cgradOutput)
                rnnlibm:accGradParameters({ hid, rnnlibinput }, rgradOutput)
                cudnnm :accGradParameters(
                    { chid, cudnninput:split(1) },
                    cgradOutput)
            end

            local rp, rg = rnnlibm:parameters()
            local cp, cg = cudnnm :parameters()

            -- Assert gradients wrt inputs are the same.
            assertGradientEq(
                rnnlibm, cudnnm, rg, cg, cellstring, bsz, allsame, i)

            if updateParams then
                rnnlibm:updateParameters(5)
                cudnnm :updateParameters(5)

                local rw = torch.cat(
                    nn.MapTable(nn.View(-1):cuda()):forward(rp), 1)
                local cw
                if allsame then
                    cw = cp[1]:narrow(1, 1, rw:size(1))
                else
                    cw = torch.cat({
                        cp[1]:narrow(1, 1, rp[1]:nElement() + rp[2]:nElement()),
                        cp[2]:narrow(1, 1, rp[3]:nElement() + rp[4]:nElement()),
                    }, 1)
                end
                tester:eq(
                    rw, cw, tol,
                    string.format("%s weight in batch %i", cellstring, i))
            end
        end
    end
end

-- The input size, hidden sizes, sequence length, batchsize, and number of batches.
local ins, nhid1, nhid2s, seqlen, bsz, nbatch = 32, 128, {128, 64}, 30, 25, 10
-- If true, copy over the hidden state from Rnnlib over to cudnn every batch.
local tieHidden    = false
-- If true, update the parameters at each batch.
local updateParams = true
-- If true, use backwards, otherwise call updateGradInput and accGradParameters.
local usebackwards = {true, false}
-- The initialization function for all tensors.
local initfun = function(x) return x:uniform(-0.01, 0.01) end

-- Generate test cases.
for _, usebackward in pairs(usebackwards) do
    -- 1 layer rnns.
    cudnntest[string.format('LSTM%s-%s', nhid1, usebackward)] = makeTest(
        rnnlib.makeRecurrent(cell.LSTM, ins, { nhid1 }):cuda(),
        { [0] = ins, [1] = nhid1 }, seqlen, bsz, nbatch, "LSTM"   ,
        tieHidden, updateParams, usebackward, initfun)
    cudnntest[string.format('GRU%s-%s', nhid1, usebackward)]  = makeTest(
        rnnlib.makeRecurrent(cell.GRU, ins, { nhid1 }):cuda(),
        { [0] = ins, [1] = nhid1 }, seqlen, bsz, nbatch, "GRU"    ,
        tieHidden, updateParams, usebackward, initfun)
    cudnntest[string.format('RNN%s-%s', nhid1, usebackward)]  = makeTest(
        rnnlib.makeRecurrent(cell.RNNTanh, ins, { nhid1 }):cuda(),
        { [0] = ins, [1] = nhid1 }, seqlen, bsz, nbatch, "RNNTanh",
        tieHidden, updateParams, usebackward, initfun)

    -- 2 layer rnns.
    for _, nhid2 in pairs(nhid2s) do
        cudnntest[string.format('LSTM%s-%s-%s', nhid1, nhid2, usebackward)] = makeTest(
            rnnlib.makeRecurrent(cell.LSTM, ins, { nhid1, nhid2 }):cuda(),
            { [0] = ins, [1] = nhid1, [2] = nhid2 }, seqlen, bsz, nbatch, "LSTM"   ,
            tieHidden, updateParams, usebackward, initfun)
        cudnntest[string.format('GRU%s-%s-%s', nhid1, nhid2, usebackward)]  = makeTest(
            rnnlib.makeRecurrent(cell.GRU, ins, { nhid1, nhid2 }):cuda(),
            { [0] = ins, [1] = nhid1, [2] = nhid2 }, seqlen, bsz, nbatch, "GRU"    ,
            tieHidden, updateParams, usebackward, initfun)
        cudnntest[string.format('RNN%s-%s-%s', nhid1, nhid2, usebackward)]  = makeTest(
            rnnlib.makeRecurrent(cell.RNNTanh, ins, { nhid1, nhid2 }):cuda(),
            { [0] = ins, [1] = nhid1, [2] = nhid2 }, seqlen, bsz, nbatch, "RNNTanh",
            tieHidden, updateParams, usebackward, initfun)
    end
end

-- Tests for the nn.RNN interface.
cudnntest[string.format('nn.LSTM%s', nhid1)] = makeTest(
    nn.LSTM{
        inputsize = ins,
        hidsize   = nhid1,
        nlayer    = 2,
        usecudnn  = false,
    }:cuda(),
    { [0] = ins, [1] = nhid1, [2] = nhid1 }, seqlen, bsz, nbatch, "LSTM",
    false, true, true, initfun)
cudnntest[string.format('nn.GRU%s', nhid1)] = makeTest(
    nn.GRU{
        inputsize = ins,
        hidsize   = nhid1,
        nlayer    = 2,
        usecudnn  = false,
    }:cuda(),
    { [0] = ins, [1] = nhid1, [2] = nhid1 }, seqlen, bsz, nbatch, "GRU",
    false, true, true, initfun)

-- Simple test cases.
function cudnntest.testSimpleRNN()
    local ins, nhid, time = 1, 1, 10

    local rnnlibm = rnnlib.makeRecurrent(cell.RNNTanh, ins, {nhid, nhid})
        :cuda()
    local p = rnnlibm:parameters()
    p[1]:fill(2 * (math.atanh(0.5) - 0.5))
    p[2]:fill(1)
    p[3]:fill(2 * (math.atanh(0.5) - 0.5))
    p[4]:fill(1)

    local hids         = {[0] = ins, [1] = nhid, [2] = nhid}
    local cudnnm       = nn.WrappedCudnnRnn(rnnlibm, "RNNTanh", hids, true)

    local hid          = rnnlibm:initializeHidden(1)
    hid[1]:fill(0.5)
    hid[2]:fill(0.5)
    local cudnninput   = torch.CudaTensor(time, 1, ins):fill(0.5)
    local rnnlibinput  = cudnninput:split(1)
    for i = 1, #rnnlibinput do
        rnnlibinput[i] = rnnlibinput[i]:view(1, -1)
    end
    rnnlibinput        = {hid, rnnlibinput}

    local cudnnHiddenInput = torch.CudaTensor(2, 1, nhid)
    cudnnHiddenInput[1]:copy(hid[1]:view(1, 1, nhid))
    cudnnHiddenInput[2]:copy(hid[2]:view(1, 1, nhid))

    cudnninput = { {cudnnHiddenInput}, cudnninput:split(1) }

    rnnlibm:cuda()
    cudnnm :cuda()

    local rnnliboutput = rnnlibm:forward(rnnlibinput)
    local cudnnoutput  = cudnnm :forward(cudnninput)

    -- Check hiddens.
    for depth = 1, 2 do
        tester:eq(
            rnnliboutput[1][depth][time],
            cudnnoutput[1][1][depth],
            tol
        )
    end
    -- Check output.
    tester:eq(rnnliboutput[2][2], cudnnoutput[2][1], tol)

    tester:eq(torch.cat(cudnnoutput[2][1]):eq(0.5):sum(), time, tol)
end

function cudnntest.testSimpleGRU()
    local ins, nhid, time = 1, 1, 1

    local rnnlibm = rnnlib.makeRecurrent(cell.GRU, ins, {nhid, nhid})
        :cuda()
    local p = rnnlibm:parameters()
    p[1]:fill(2 * (math.atanh(0.5) - 0.5))
    p[2]:fill(1)
    p[3]:fill(2 * (math.atanh(0.5) - 0.5))
    p[4]:fill(1)

    local hids         = {[0] = ins, [1] = nhid, [2] = nhid}
    local cudnnm       = nn.WrappedCudnnRnn(rnnlibm, "GRU", hids, true)

    local hid          = rnnlibm:initializeHidden(1)
    hid[1]:fill(0.5)
    hid[2]:fill(0.5)
    local cudnninput   = torch.CudaTensor(time, 1, ins):fill(0.5)
    local rnnlibinput  = cudnninput:split(1)
    for i              = 1, #rnnlibinput do
        rnnlibinput[i] = rnnlibinput[i]:view(1, -1)
    end
    rnnlibinput        = { hid, rnnlibinput }

    local cudnnHiddenInput = torch.CudaTensor(2, 1, nhid)
    cudnnHiddenInput[1]:copy(hid[1]:view(1, 1, nhid))
    cudnnHiddenInput[2]:copy(hid[2]:view(1, 1, nhid))

    cudnninput = { {cudnnHiddenInput}, cudnninput:split(1) }

    rnnlibm:cuda()
    cudnnm :cuda()

    local rnnliboutput = rnnlibm:forward(rnnlibinput)
    local cudnnoutput  = cudnnm :forward(cudnninput)

    -- Check hiddens.
    for depth = 1, 2 do
        tester:eq(rnnliboutput[1][depth][time],
                  cudnnoutput[1][1][depth], tol)
    end
    -- Check output.
    tester:eq(rnnliboutput[2][2], cudnnoutput[2][1], tol)
end

tester:add(cudnntest)
tester:run()
