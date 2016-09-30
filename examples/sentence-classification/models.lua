-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Models for SST-1 Sentiment Classification
--------------------------------------------------------------------------------

local rnnlib   = require 'rnnlib'
local mutils   = require 'rnnlib.mutils'

local models = {}

-- | A simple forward RNN for encoding sentences.
function models.rnn(ntoken, hids, nclasses, cell)
    local init_range = 0.1
    local emsize = hids[0]
    local lut = nn.LookupTable(ntoken, emsize)
    lut.weight:uniform(-init_range, init_range)
    local rnn = rnnlib.makeRecurrent{
        cellfn     = cell,
        inputsize  = emsize,
        hids       = hids,
        savehidden = false,
    }
    local decoder = nn.Linear(hids[#hids], nclasses)
    decoder.bias:fill(0)
    decoder.weight:uniform(-init_range, init_range)

    return nn.Sequential()
        :add(mutils.batchedinmodule(rnn, lut)) -- Lookup table + RNN
        :add(nn.SelectTable(2))                -- Select RNN output
        :add(nn.SelectTable(-1))               -- Select last layer in RNN
        :add(nn.MapTable(nn.View(1, -1, hids[#hids])))
        :add(nn.JoinTable(1))                  -- Join all timesteps together
        :add(nn.Sum(1, 3, true))               -- Average embeddings
        :add(decoder)
    , rnn
end

-- | A bidirectional RNN that uses a mask to discard padding information
-- to enable batching.
-- ntoken      : The number of tokens in the input vocabulary.
-- hids        : A table representating the hidden units at each layer.
-- nclasses    : The number of classes.
-- sharefields : A table of the fields to be shared between the fwd and rev RNNs.
-- dropout     : A number representing the dropout parameter.
function models.birnn(ntoken, hids, nclasses, cell, sharefields, dropout)
    local init_range = 0.1
    local emsize = hids[0]
    local lut = nn.LookupTable(ntoken, emsize)
    lut.weight:uniform(-init_range, init_range)

    -- This will not save the hidden state in between batches.
    local birnn = rnnlib.makeBidirectional(
        cell,
        emsize,
        hids,
        sharefields
    )

    local decoder = nn.Linear(2 * hids[#hids], nclasses)
    decoder.bias:fill(0)
    decoder.weight:uniform(-init_range, init_range)

    local rawinput = nn.Identity()()
    local input, mask = rawinput:split(2)
    local rnn = nn.Sequential()
        :add(mutils.batchedinmodule(birnn, lut)) -- Lookup table + RNN
        :add(nn.SelectTable(2))                  -- Select RNN output
        :add(nn.SelectTable(-1))                 -- Select last layer in RNN
        :add(nn.MapTable(nn.View(1, -1, 2 * hids[#hids])))
        :add(nn.JoinTable(1))                    -- Join all timesteps together
        (input)

    local rnnout = nn.CMulTable(){rnn, mask}
    local reduced = dropout > 0
        and nn.Dropout(dropout)(nn.Sum(1, 3, false)(rnnout))
        or  nn.Sum(1, 3, false)(rnnout)

    return nn.gModule({ rawinput }, { decoder(reduced) })
    , birnn
end

return models
