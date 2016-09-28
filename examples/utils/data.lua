-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Utilities for tokenization
--------------------------------------------------------------------------------

local argcheck = require 'argcheck'
local tds      = require 'tds'
local pl       = require('pl.import_into')()

-- Setup timer.
local timer = torch.Timer()
-- | Start timing section.
local function tic()
    timer:reset()
end
-- | End timing section.
local function toc()
    return timer:time().real
end

local data = {}

-- | Trim whitespace.
local function trim(s)
    local n = s:find('%S')
    return n and s:match('.*%S', n) or ''
end

-- | Add a word to a dictionary.
data.addword = function(dic, word)
    if dic.word2idx[word] == nil then
        dic.idx2word:insert(word)
        dic.word2idx[word] = #dic.idx2word
        dic.idx2freq[#dic.idx2word] = 1
    else
        dic.idx2freq[dic.word2idx[word]] =
            dic.idx2freq[dic.word2idx[word]] + 1
    end
    return dic
end

-- Words common to all corpuses.
-- You can add <pad>.
local default_words = {
    "</s>", "<unk>",
}

-- | Initialize a dictionary as a table.
-- dic.idx2freq will be fed into a tensor constructor.
data.initdictionary = function(common_words)
    common_words = common_words or default_words
    local dic = {
        idx2word = tds.Vec(),
        word2idx = tds.Hash(),
        idx2freq = {},
    }

    for _, w in pairs(common_words) do
        data.addword(dic, w)
    end

    return dic
end

-- | Create a dictionary from a file without filtering.
data.makedictionary = argcheck{
    { name = 'filename'     , type = 'string' ,                         },
    { name = 'common_words' , type = 'table'  , default = default_words },
    call = function(filename)
        tic()
        local dic = data.initdictionary()
        local lines = 0
        local gccounter = 0
        for line in io.lines(filename) do
            local sline = pl.stringx.split(line, ' ')
            for _, w in ipairs(sline) do
                w = w:gsub('%s*', '')
                if w ~= '' then
                    data.addword(dic, w)
                    gccounter = gccounter + 1
                end
            end
            lines = lines + 1

            if gccounter > 1000000 then
                collectgarbage()
                gccounter = 0
            end
        end
        dic.idx2freq[dic.word2idx['</s>']] = lines
        dic.idx2freq[dic.word2idx['<unk>']] = dic.idx2freq[2] ~= 0
            and dic.idx2freq[2] or 1 -- nonzero hack

        dic.idx2freq = torch.Tensor(dic.idx2freq)
        print(string.format('| Dictionary size %d', #dic.idx2word))
        print('* Dictionary took', toc(), 's')
        return dic
    end
}

-- | Rebuild a dictionary given the new indices and frequencies.
local function remakedic(dic, idxs, freqs)
    local new_dic = data.initdictionary()
    new_dic.idx2freq = freqs
    for i = 1, idxs:size(1) do
        local word = dic.idx2word[idxs[i]]
        new_dic.idx2word:insert(word)
        new_dic.word2idx[word] = #new_dic.idx2word
    end
    print(string.format('| Pruned dictionary size %d', #new_dic.idx2word))
    return new_dic
end

-- A table of functions with different strategies for pruning dictionaries.
data.prune = {
    -- | Filter based on a frequency threshold.
    threshold = function(idx2freq, value)
        -- Seems like nonzero has a bug: 1d vector returns 2d.
        local idxs  = idx2freq:gt(value):nonzero():squeeze()
        local freqs = idx2freq:index(1, idxs)
        return idxs, freqs
    end,
    -- | Take the top `value` occurring words.
    histogram = function(idx2freq, value)
        local freqs, idxs = torch.sort(idx2freq, true)
        return idxs:narrow(1, 1, value), freqs:narrow(1, 1, value)
    end,
}

-- | Prune a dictionary.
-- Prune should be a function from the above data.prune table.
data.prunedictionary = argcheck{
    { name = 'dic'          , type = 'table'    ,                         },
    { name = 'prune'        , type = 'function' ,                         },
    { name = 'value'        , type = 'number'   ,                         },
    { name = 'common_words' , type = 'table'    , default = default_words },
    call = function(dic, prune, value, common_words)
        local word_store = {}
        for _, w in pairs(common_words) do
            -- zero these out since they will be added in remakedic
            word_store[w] = dic.idx2freq[dic.word2idx[w]]
            dic.idx2freq[dic.word2idx[w]] = 0
        end

        local newdic = remakedic(dic, prune(dic.idx2freq, value))

        for _, w in pairs(common_words) do
            -- reset back to original values
            newdic.idx2freq[newdic.word2idx[w]] = word_store[w]
        end

        return newdic
    end
}

-- | Tokenize a file using a dictionary.
data.loadFile = argcheck{
    { name = 'filename'      , type = 'string'  ,                 },
    { name = 'dic'           , type = 'table'   ,                 },
    { name = 'preservelines' , type = 'boolean' , default = false },
    call = function(filename, dic, preservelines)
        tic()
        local words     = 0
        local lines     = 0
        local gccounter = 0

        -- Count # of words in corpus.
        for line in io.lines(filename) do
            local _, occurrences = trim(line):gsub('%s+', '')
            _ = nil
            -- +1 because of undercounting of separaters.
            -- +1 because of eos.
            words = words + occurrences + 2
            lines = lines + 1

            gccounter = gccounter + occurrences
            if gccounter > 1000000 then
                collectgarbage()
                gccounter = 0
            end
        end
        collectgarbage()
        collectgarbage()

        local data = preservelines == true
            and {}
            or  torch.LongTensor(words):fill(0)

        words     = 0
        lines     = 0
        gccounter = 0

        -- Collect tokens.
        for line in io.lines(filename) do
            lines = lines + 1
            -- This is a memory killer: sometimes works, sometimes doesn't,
            -- depending on the line size.
            local sline   = pl.stringx.split(trim(line):gsub('%s+', ' '), ' ')
            line          = nil
            local linelen = #sline

            if preservelines then
                data[lines] = torch.LongTensor(linelen+1)
            end

            for i = 1, linelen do
                local w  = sline[i]:gsub('%s*', '')
                sline[i] = nil

                if w ~= '' then
                    words = words + 1
                    if preservelines then
                        data[lines][i] = dic.word2idx[w] or dic.word2idx['<unk>']
                    else
                        data[words] = dic.word2idx[w] or dic.word2idx['<unk>']
                    end
                    gccounter = gccounter + 1
                    if gccounter >= 1e6 then
                        collectgarbage()
                        gccounter = 0
                    end
                end
            end

            -- Insert the eos </s> character.
            words       = words + 1
            if preservelines then
                data[lines][linelen+1] = dic.word2idx['</s>']
            else
                data[words] = dic.word2idx['</s>']
            end
        end

        local numtokens
        if preservelines then
            numtokens = 0
            for i = 1, #data do
                numtokens = numtokens + data[i]:nElement()
            end
        else
            numtokens = data:size(1)
        end

        print(string.format(
            "| Load file %s: %d tokens", filename,
            preservelines and #data or data:size(1)))
        collectgarbage()
        print('* Load file took: ', toc(), 's')

        return data
    end
}

data.formatData = function(data, bsz, bptt, val, policy)
    -- Batches are of size bptt x batchsize.
    -- The data needs to be transform into a tensor of
    -- size nbatch x bptt x batchsize.

    val = val or 1
    policy = policy or 'pad-back'
    local ntoken = data:nElement()

    -- Round the number of batches down if not padding.
    local nbatch = policy == "pad-none"
        and math.floor(ntoken / (bsz*bptt))
        or  math.ceil (ntoken / (bsz*bptt))

    local tsize = nbatch * bsz * bptt
    local ndata = torch.Tensor(tsize):typeAs(data):fill(val)

    if policy == 'pad-back' then
        ndata:narrow(1, 1, ntoken):copy(data)
    elseif policy == 'pad-front' then
        ndata:narrow(1, tsize - ntoken + 1, ntoken):copy(data)
    elseif policy == 'pad-none' then
        ndata:copy(data:narrow(1, 1, tsize))
    else
        error('invalid padding policy')
    end

    data = ndata:view(bsz, nbatch, bptt):permute(2,3,1)
    return data
end

-- | Batched padding options.
-- buffer    : torch.Tensor(batchsize, padlength) where the result will be placed.
-- sentences : A table of the sentence sin torch.Tensor form.
-- padlength : The length of the buffer's second dimension.
-- padvalue  : The value of the padding token.
data.batchpad = {
    -- Place padding tokens in the front.
    front = function(buffer, sentences, padlength, padvalue)
        buffer:resize(padlength, #sentences):fill(padvalue)
        for i = 1, #sentences do
            local len = sentences[i]:size(1)
            buffer
                :select(2, i)
                :narrow(1, padlength - len + 1, len)
                :copy(sentences[i])
        end
    end,
    -- Place padding tokens in the back.
    back  = function(buffer, sentences, padlength, padvalue)
        buffer:resize(padlength, #sentences):fill(padvalue)
        for i = 1, #sentences do
            local len = sentences[i]:size(1)
            buffer
                :select(2, i)
                :narrow(1, 1, len)
                :copy(sentences[i])
        end
    end,
    -- Place padding tokens so that each sentence is sort of in the middle.
    both  = function(buffer, sentences, padlength, padvalue)
        buffer:resize(padlength, #sentences):fill(padvalue)
        for i = 1, #sentences do
            local len = sentences[i]:size(1)
            buffer
                :select(2, i)
                :narrow(1, math.floor((padlength - len) / 2) + 1, len)
                :copy(sentences[i])
        end
    end,
}

return data
