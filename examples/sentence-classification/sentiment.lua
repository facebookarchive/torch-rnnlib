-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- SST-1 Sentence Classification
--   A sentence classification example using the Stanford Sentiment Treebank.
-- The hidden state is not passed in between batches for the RNN because
-- each example is independent from every other.
--------------------------------------------------------------------------------

local basepath = 'rnnlib.examples.sentence-classification.'

local argcheck = require 'argcheck'
local rnnlib   = require 'rnnlib'
local dutils   = require 'rnnlib.examples.utils.data'

local cmd = torch.CmdLine()
-- * Model options.
cmd:option('-model'          , 'birnn' , 'The model type string.'                                        )
cmd:option('-cell'           , 'LSTM'  , 'The model type string.'                                        )
cmd:option('-insize'         , 256     , 'The number of hidden units in the lookup table.'               )
cmd:option('-nhid'           , 512     , 'The number of hidden units per layer in the RNN.'              )
cmd:option('-nlayer'         , 2       , 'The number of layers in the RNN.'                              )
cmd:option('-dropout'        , 0       , 'The probability of dropout.'                                   )
-- * Optimization options.
cmd:option('-lr'             , 1       , 'The learning rate.'                                            )
cmd:option('-clip'           , 0       , 'The clip threshold of the norm of the gradients w.r.t. params.')
cmd:option('-batchsize'      , 20      , 'The batch size.'                                               )
-- * Training options.
cmd:option('-maxepoch'       , 10      , 'The upper epoch limit.'                                        )
cmd:option('-profbatch'      , 0       , 'The number of batches for profiling.'                          )
cmd:option('-reportinterval' , 1000    , 'The number of batches after which to report.'                  )
-- * More model options.
cmd:option('-share'          , false   , 'Share the parameters of the fwd + rev RNN.'                    )
cmd:option('-cudnn'          , false   , 'Use cudnn for the RNN model.'                                  )
-- * Misc options.
cmd:option('-devid'          , 1       , 'The master device id.'                                         )
cmd:option('-seed'           , 1111    , 'The random seed for the CPU + GPU.'                            )
cmd:option('-save'           , ''      , 'Save the model at the end of training.'                        )
local config = cmd:parse(arg)

local usegpu = config.devid > 0

torch.manualSeed(config.seed)
if usegpu then
    pcall(require, 'cutorch')
    assert(cutorch)
    cutorch.manualSeed(config.seed)
    cutorch.setDevice (config.devid)
end

local function printf(...) print(string.format(...)) end

--------------------------------------------------------------------------------
-- Data loading
--------------------------------------------------------------------------------

-- The directory where the data is either created or is already stored.
local datadir = '/tmp/sst1'

local insz = config.insize
local nhid = config.nhid
local bsz  = config.batchsize
local clip = config.clip
local lr   = config.lr

local reportinterval = config.reportinterval

-- This is hardcoded based on the dataset.
local nclasses = 5

local batches, dict = require(basepath .. 'get_data')(datadir)

local train = batches.train
local valid = batches.valid
local test  = batches.test

--------------------------------------------------------------------------------
-- Model construction
--------------------------------------------------------------------------------

local hids = {}
hids[0] = insz
for i = 1, config.nlayer do
    hids[i] = nhid
end

-- Creates a forward or bidirectional RNN.
-- See models.lua for more information.
local model, rnn = require(basepath .. 'models')[config.model](
    #dict.idx2word,
    hids,
    nclasses,
    rnnlib.cell[config.cell],
    config.share and { 'weight', 'gradWeight', 'bias', 'gradBias' } or nil,
    config.dropout)

local criterion = nn.CrossEntropyCriterion()

if usegpu then
    model    :cuda()
    criterion:cuda()
end

local _, grads = model:parameters()

--------------------------------------------------------------------------------
-- Training utility functions
--------------------------------------------------------------------------------

-- | A function to create examples for bidirectional models.
-- No new memory is allocated.
local getexamplebidirectional = argcheck{
    noordered = true,
    { name = "data"         , type = "table"         ,            },
    { name = "i"            , type = "number"        ,            },
    { name = "bsz"          , type = "number"        ,            },
    { name = "inputbuffer"  , type = "torch.*Tensor" ,            },
    { name = "maskbuffer"   , type = "torch.*Tensor" ,            },
    { name = "targetbuffer" , type = "torch.*Tensor" ,            },
    { name = "perm"         , type = "torch.*Tensor" , opt = true },
    call = function(data, i, bsz, inputbuffer, maskbuffer, targetbuffer, perm)
        if i + bsz - 1 > #data then
            bsz = #data - i + 1
        end
        -- Gather examples.
        local inputs  = {}
        local targets = {}
        local maxlen  = 0
        for idx = i, i + bsz - 1 do
            idx = perm and perm[idx] or idx
            -- Populate example.
            table.insert(inputs,  data[idx].input)
            table.insert(targets, data[idx].target)
            -- Find the max length.
            if data[idx].input:size(1) > maxlen then
                maxlen = data[idx].input:size(1)
            end
        end
        -- Prepare masks. Padding is always put in front of the sentence.
        maskbuffer:resize(maxlen, bsz, 1)
        -- Zero out activations for padded elements.
        maskbuffer:fill  (0)
        for i = 1, #inputs do
            local len = inputs[i]:size(1)
            -- Fill in the last indices with 1's to keep activations alive.
            maskbuffer:select(2, i):narrow(1, maxlen - len + 1, len):fill(1)
        end
        local lengths = maskbuffer:sum(1):repeatTensor(maxlen, 1, 1)
        -- Normalize by length.
        maskbuffer:cdiv(lengths)
        -- Expand to hidden dimension.
        maskbuffer = maskbuffer:expand(maxlen, bsz, nhid * 2)
        -- Pad input.
        dutils.batchpad.front(inputbuffer, inputs, maxlen, 0)
        -- Fill in targets.
        targetbuffer:resize(bsz)
        for i = 1, #targets do
            targetbuffer[i] = targets[i]
        end
        return  { tokens = inputbuffer, mask = maskbuffer }, targetbuffer
    end,
}

-- | Creates examples for a forward RNN.
local getexampleforward = argcheck{
    noordered = true,
    { name = "data"         , type = "table"         ,            },
    { name = "i"            , type = "number"        ,            },
    { name = "bsz"          , type = "number"        ,            },
    { name = "inputbuffer"  , type = "torch.*Tensor" ,            },
    { name = "targetbuffer" , type = "torch.*Tensor" ,            },
    { name = "perm"         , type = "torch.*Tensor" , opt = true },
    call = function(data, i, bsz, inputbuffer, targetbuffer, perm)
        -- Gather examples.
        local inputs  = {}
        local targets = {}
        local maxlen  = 0
        for idx = i, i + bsz do
            idx = perm and perm[idx] or idx
            -- Populate example.
            table.insert(inputs,  data[idx].input)
            table.insert(targets, data[idx].target)
            -- Find the max length.
            if data[idx]:size(1) > maxlen then
                maxlen = data[idx]:size(1)
            end
        end
        dutils.batchpad.front(inputbuffer, inputs, maxlen, 0)
        targetbuffer:resize(bsz)
        for i = 1, #targets do
            targetbuffer[i] = targets[i]
        end
        return { tokens = inputbuffer }, targetbuffer
    end,
}

-- Use bidirection or forward example getters.
local getexample = config.model:find('bi')
    and getexamplebidirectional
    or  getexampleforward

-- | Perform the forward pass.
local function forward(model, input, target, criterion)
    return criterion:forward(
        model:forward{ { rnn.hiddenbuffer, input.tokens }, input.mask },
        target
    )
end

-- | Perform the backward pass.
local function backward(model, input, target, criterion)
    model:zeroGradParameters()
    model:backward(
        { { rnn.hiddenbuffer, input.tokens }, input.mask  },
        criterion:backward(model.output, target)
    )
end

-- | Clip the gradients to prevent explosion.
local function clipgradients(grads, norm)
    local totalnorm = 0
    for mm = 1, #grads do
        local modulenorm = grads[mm]:norm()
        totalnorm = totalnorm + modulenorm * modulenorm
    end
    totalnorm = math.sqrt(totalnorm)
    if totalnorm > norm then
        local coeff = norm / math.max(totalnorm, 1e-6)
        for mm = 1, #grads do
            grads[mm]:mul(coeff)
        end
    end
end

-- | Compute the class accuracy given the model output and target vector.
local function classacc(output, target)
    assert(output:nDimension() == 2)
    assert(target:nDimension() == 1)
    local no = output:size(1)
    local _, pred = output:double():topk(1, 2, true, true)
    local correct = pred
        :typeAs(target)
        :eq(target:view(no, 1):expandAs(pred))
    return correct:sum(), no
end

-- | Evaluate the model on some part of the data.
local function evaluate(model, data, bsz, criterion, buffers, perm)
    model:evaluate()
    local loss = 0
    local numexamples = 0
    local ncorrect = 0
    local noutputs = 0
    -- Loop over data.
    for i = 1, #data, bsz do
        local input, target = getexample{
            data         = data,
            i            = i,
            bsz          = bsz,
            inputbuffer  = buffers.input,
            maskbuffer   = buffers.mask,
            targetbuffer = buffers.target,
            perm         = perm,
        }

        rnn:initializeHidden(target:size(1))
        loss = loss + forward(model, input, target, criterion)
        numexamples = numexamples + 1

        local nc, no = classacc(model.output, target)
        ncorrect = ncorrect + nc
        noutputs = noutputs + no
    end
    -- Average out the loss.
    return loss / numexamples, ncorrect, noutputs
end

--------------------------------------------------------------------------------
-- Training
--------------------------------------------------------------------------------

-- Set up auxiliary tools and  buffers for training.
local timer = torch.Timer()
local inputbuffer  = usegpu and torch.CudaLongTensor() or torch.LongTensor()
local targetbuffer = usegpu and torch.CudaTensor()     or torch.LongTensor()
local maskbuffer
if config.model:find('bi') then
    maskbuffer = usegpu and torch.CudaTensor() or torch.LongTensor()
end

local buffers = {
    input  = inputbuffer,
    mask   = maskbuffer,
    target = targetbuffer,
}

local prevval
for epoch = 1, config.maxepoch do
    model:training()
    local trainperm   = torch.randperm(#train)
    local loss        = 0
    local numexamples = 0
    local ncorrect    = 0
    local noutputs    = 0
    timer:reset()
    for i = 1, #train, bsz do
        local input, target = getexample{
            data         = train,
            i            = i,
            bsz          = bsz,
            inputbuffer  = inputbuffer,
            maskbuffer   = maskbuffer,
            targetbuffer = targetbuffer,
            perm         = trainperm,
        }
        -- Re-initializing the hidden unit is not necessary for the most part,
        -- since the models do not save the hidden state.
        -- However, this is useful for resizing the initial hidden state.
        rnn:initializeHidden(target:size(1))
        loss = loss + forward(model, input, target, criterion)
        backward(model, input, target, criterion)
        if clip > 0 then clipgradients(grads, clip) end
        model:updateParameters(lr)

        -- Accumulate the class accuracy.
        local nc, no = classacc(model.output, target)
        ncorrect = ncorrect + nc
        noutputs = noutputs + no

        numexamples = numexamples + 1
        if numexamples % reportinterval == 0 then
            local trainloss = loss / numexamples
            printf(
                '| epoch %03d | %05d samples | lr %02.6f | ms/batch %3d | '
                    .. 'train loss %5.2f | train class acc %0.4f',
                epoch, numexamples, lr, timer:time().real * 1000 / numexamples,
                trainloss, ncorrect / noutputs
            )
        end
    end

    -- Perform validation.
    loss, ncorrect, noutputs = evaluate(model, valid, bsz, criterion, buffers)
    if prevval and loss > prevval then lr = lr / 2 end
    prevval = loss

    printf(
        '| end of epoch %03d | ms/batch %7d | '
            .. 'valid loss %5.2f | valid class acc %0.3f',
        epoch, timer:time().real * 1000 / numexamples,
        loss, ncorrect / noutputs
    )

    if lr < 1e-5 then break end
end

-- Evaluate on test set.
local loss, ncorrect, noutputs = evaluate(model, test, bsz, criterion, buffers)
printf(
    "| End of training | test loss %5.2f | test class acc %0.3f",
    loss, ncorrect / noutputs
)

if config.save ~= '' then
    torch.save(config.save, model)
end
