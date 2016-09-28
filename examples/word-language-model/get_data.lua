-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Data Fetching Script for PTB
--------------------------------------------------------------------------------

require 'sys'
require 'paths'

local stringx  = require('pl.import_into')().stringx

-- | Initialize a very basic dictionary object.
local function initdictionary()
    return {
        idx2word = {},
        word2idx = {},
    }
end

local function addword(dict, word)
    if not dict.word2idx[word] then
        local id = #dict.idx2word + 1
        dict.word2idx[word] = id
        dict.idx2word[id]   = word
        return id
    end
    return dict.word2idx[word]
end

-- | Tokenize a text file.
local function loadfile(path, dict)
    -- Read words from file.
    assert(paths.filep(path))
    local tokens = 0
    for line in io.lines(path) do
        local words = stringx.split(
            line:gsub("%s+", " "):gsub("^%s*", ""):gsub("%s*$", ""),
            " "
        )
        for _, word in ipairs(words) do
            addword(dict, word)
            tokens = tokens + 1
        end
        addword(dict, "<eos>")
        tokens = tokens + 1
    end

    local ids = torch.LongTensor(tokens)
    local token = 1
    for line in io.lines(path) do
        local words = stringx.split(
            line:gsub("%s+", " "):gsub("^%s*", ""):gsub("%s*$", ""),
            " "
        )
        for _, word in ipairs(words) do
            ids[token] = dict.word2idx[word]
            token = token + 1
        end
        ids[token] = dict.word2idx['<eos>']
        token = token + 1
    end


    -- Final dataset.
    return ids
end

return function(path)
    path = path or "."
    local dict = initdictionary()
    return {
        train = loadfile(path .. '/penn/train.txt', dict),
        valid = loadfile(path .. '/penn/valid.txt', dict),
        test  = loadfile(path .. '/penn/test.txt',  dict),
        dict  = dict,
    }
end
