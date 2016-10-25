-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Language Modeling on Penn Tree Bank
--------------------------------------------------------------------------------

require 'math'
require 'torch'
require 'rnnlib'

local mutils = require 'rnnlib.mutils'

local basepath = 'rnnlib.examples.word-language-model.'

local cmd = torch.CmdLine()
-- * Data parameters.
cmd:option( '-train'     , ''     , 'Train file.'                               )
cmd:option( '-valid'     , ''     , 'Valid file.'                               )
cmd:option( '-test'      , ''     , 'Test file.'                                )
-- * Model parameters.
cmd:option( '-model'     , 'LSTM' , 'Type of recurrent net: RNN, LSTM, or GRU.' )
cmd:option( '-emsize'    , 200    , 'Number of hidden units per layer.'         )
cmd:option( '-nhid1'     , 200    , 'Number of hidden units per layer.'         )
cmd:option( '-nhid2'     , 200    , 'Number of hidden units per layer.'         )
-- * Optimization parameters.
cmd:option( '-lr'        , 20     , 'Initial learning rate.'                    )
cmd:option( '-clip'      , 0.5    , 'Gradient clipping.'                        )
cmd:option( '-maxepoch'  , 6      , 'Upper epoch limit.'                        )
cmd:option( '-batchsize' , 20     , 'Batch size.'                               )
cmd:option( '-bptt'      , 20     , 'Sequence length.'                          )
-- * Device parameters.
cmd:option( '-devid'     , 0      , 'GPU device id.'                            )
cmd:option( '-seed'      , 1111   , 'Random seed.'                              )
cmd:option( '-cudnn'     , false  , 'Use cudnn for speeding up the RNN.'        )
-- * Misc parameters.
cmd:option( '-reportint' , 1000   , 'Report interval.'                          )
cmd:option( '-save'      , 'm.t7' , 'Path to save the final model.'             )
local config = cmd:parse(arg)

-- Set the random seed manually for reproducibility.
torch.manualSeed(config.seed)
-- If the GPU is enabled, do some plumbing.
local usegpu = config.devid > 0
if usegpu then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice (config.devid)
    cutorch.manualSeed(config.seed)
end
if config.cudnn then
    assert(usegpu, "Please specify the device id.")
    require 'cudnn'
end

local function printf(...) print(string.format(...)) end

--------------------------------------------------------------------------------
-- LOAD DATA
--------------------------------------------------------------------------------

local batches = require(basepath .. 'get_data'){
    train = config.train,
    valid = config.valid,
    test  = config.test,
}
local dict  = batches.dict
local train = batches.train
local valid = batches.valid
local test  = batches.test

torch.save('batches.t7', batches)

-- Truncate train and reshape.
local bptt   = config.bptt
local bsz    = config.batchsize
local nbatch = math.floor(train:size(1) / bptt / bsz)
train = train:narrow(1, 1, nbatch * bptt * bsz)

-- Ensure that valid and test are divisible by bsz * bptt.
local validbsz, testbsz = 10, 10
local validbptt, testbptt = 1, 1

-- Divide up into batches.
train = train:view(bsz,      -1):t():contiguous()
valid = valid:view(validbsz, -1):t():contiguous()
test  = test :view(testbsz,  -1):t():contiguous()

collectgarbage()

--------------------------------------------------------------------------------
-- MAKE MODEL
--------------------------------------------------------------------------------

local initrange = 0.1

-- Size of the vocabulary.
local ntoken  = #dict.idx2word

local hids = {
    [0] = config.emsize,
    [1] = config.nhid1,
    [2] = config.nhid2 > 0 and config.nhid2 or nil
}

local lut = nn.LookupTable(ntoken, hids[0])
lut.weight:uniform(-initrange, initrange)

local rnn = nn[config.model]{
    inputsize = hids[0],
    hidsize   = hids[1],
    nlayer    = #hids,
    usecudnn  = config.cudnn,
}

local decoder = nn.Linear(
    config.nhid2 > 0 and config.nhid2 or config.nhid1, ntoken)
decoder.bias:fill(0)
decoder.weight:uniform(-initrange, initrange)

local model = nn.Sequential()
    :add(mutils.batchedinmodule(rnn, lut))
    -- Select the output of the RNN.
    -- The RNN's forward gives a table of { hiddens, outputs }.
    :add(nn.SelectTable(2))
    -- Select the output of the last layer, since the output
    -- of all layers are returned.
    :add(nn.SelectTable(-1))
    -- Flatten the output from bptt x bsz x ntoken to bptt * bsz x ntoken.
    -- Note that the first dimension is actually a table, so there is
    -- copying involved during the flattening.
    :add(nn.JoinTable(1))
    :add(decoder)

local criterion = nn.CrossEntropyCriterion()

-- Unroll the rnns.
if not config.cudnn then
    for i = 1, #rnn.modules do
        rnn.modules[i]:extend(bptt)
    end
end

-- If the GPU is enabled, move everything to GPU.
if usegpu then
    model:cuda()
    criterion:cuda()
end

-- Get the parameters for gradient clipping.
local _, grads = model:parameters()

--------------------------------------------------------------------------------
-- TRAINING
--------------------------------------------------------------------------------

local timer = torch.Timer()

-- Put the table values into the register file for faster access.
local lr   = config.lr
local clip = config.clip

local reportinterval = config.reportint

-- Create buffers for all tensors needed during training on GPU.
local gpubuffer
if usegpu then
    gpubuffer = torch.CudaTensor()
end

-- | Indexes into a dataset and grabs the input and target tokens.
-- Minimizes gpu memory.
local function getexample(data, i, bptt, gpubuffer)
    local newinput, newtarget
    if gpubuffer then
        gpubuffer
            :resize(bptt+1, data:size(2))
            :copy(data:narrow( 1, i, bptt+1 ))
        newinput  = gpubuffer:narrow(1, 1, bptt)
        newtarget = gpubuffer:narrow(1, 2, bptt)
    else
        newinput  = data:narrow( 1 , i    , bptt)
        newtarget = data:narrow( 1 , i + 1, bptt)
    end
    return newinput, newtarget:view(-1)
end

-- | Performs the forward computation of the model.
-- model     : The nn.Module.
-- input     : The input tokens (bptt x bsz).
-- target    : The target tokens (bptt x bsz).
-- criterion : The nn.CrossEntropyCriterion module.
local function forward(model, input, target, criterion)
    return criterion:forward(
        model:forward{ rnn.hiddenbuffer, input },
        target
    )
end

-- | Perform the backward pass.
-- model     : The nn.Module.
-- input     : (bptt x bsz).
-- target    : The target tokens (bptt x bsz).
-- criterion : The nn.CrossEntropyCriterion module.
local function backward(model, input, target, criterion)
    model:zeroGradParameters()
    model:backward(
        { rnn.hiddenbuffer, input },
        criterion:backward(model.output, target)
    )
end

-- | Gradient clipping to try to prevent the gradient from exploding.
-- Does a global in-place clip.
-- grads : The table of model gradients from :parameters().
-- norm  : The max gradient norm we rescale to.
local function clipGradients(grads, norm)
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

-- | Perform the forward pass only.
-- model     : The nn.Module.
-- data      : The data tensor (bsz, -1).
-- bptt      : The sequence length (number).
-- bsz       : The size of the batch (number).
-- criterion : The nn.CrossEntropyCriterion module.
local function evaluate(model, data, bptt, bsz, criterion)
    model:evaluate()
    local loss = 0
    local numexamples = 0
    rnn:initializeHidden(bsz)
    -- Loop over validation data.
    for i = 1, data:size(1) - bptt, bptt do
        local input, target = getexample(data, i, bptt, gpubuffer)
        loss = loss + forward(model, input, target, criterion)
        numexamples = numexamples + 1
    end
    -- Average out the loss.
    return loss / numexamples
end

-- Loop over epochs.
local prevval
for epoch = 1, config.maxepoch do
    local loss = 0
    local numexamples = 0
    timer:reset()
    -- Reset the hidden state.
    rnn:initializeHidden(bsz)
    -- Loop over the training data.
    model:training()
    for i = 1, train:size(1) - bptt, bptt do
        local input, target = getexample(train, i, bptt, gpubuffer)
        loss = loss + forward(model, input, target, criterion)
        backward(model, input, target, criterion)
        clipGradients(grads, clip)
        model:updateParameters(lr)

        numexamples = numexamples + 1
        if numexamples % reportinterval == 0 then
            local trainloss = loss / numexamples
            printf(
                '| epoch %03d | %05d samples | lr %02.6f | ms/batch %3d | '
                    .. 'train loss %5.2f | train ppl %8.2f',
                epoch, numexamples, lr, timer:time().real * 1000 / numexamples,
                trainloss, math.exp(trainloss)
            )
        end
    end

    loss = evaluate(model, valid, validbptt, validbsz, criterion)

    -- The annealing schedule.
    if prevval and loss > prevval then lr = lr / 4 end
    prevval = loss

    printf(
        "| end of epoch %03d | ms/batch %7d | valid loss %5.2f | valid ppl %8.2f",
        epoch, timer:time().real * 1000 / numexamples, loss, math.exp(loss)
    )
end

-- Run on test data.
local loss = evaluate(model, test, testbptt, testbsz, criterion)
printf(
    "| End of training | test loss %5.2f | test ppl %8.2f",
    loss, math.exp(loss)
)

if config.save ~= '' then
    model.rnn = rnn
    torch.save(config.save, model)
end
