-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Sentence Generation Script
--------------------------------------------------------------------------------

require 'rnnlib'

-- Load pretrained model and dictionary from word_lm.lua.
local model = torch.load('m.t7')
local rnn   = model.rnn
local dict  = torch.load('batches.t7').dict

local maxlength = 100
local eos       = dict.word2idx['<eos>']

rnn:initializeHidden(1)
local index = torch.CudaTensor(1, 1)
local lsm   = nn.LogSoftMax():cuda()

local words = {}
index:fill(eos)
for i = 1, maxlength do
    local output = lsm:forward(model:forward{ rnn.hiddenbuffer, index })
    torch.multinomial(index, output, 1)
    words[i] = index:sum()
    if words[i] == eos then break end
end

for k, v in pairs(words) do
    words[k] = dict.idx2word[v]
end
print(table.concat(words, ' '))
