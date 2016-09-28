-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE-examples file in the root directory of this source tree.

--------------------------------------------------------------------------------
-- Data loader for SST-1 Sentiment Classification Dataset
--------------------------------------------------------------------------------

require 'os'
require 'sys'
require 'paths'

local data = require 'rnnlib.examples.utils.data'

-- | Load labels from a file.
local function getLabels(filepath)
    local labels = {}
    for line in io.lines(filepath) do
        table.insert(labels, tonumber(line) + 1)
    end
    return torch.LongTensor(labels)
end

-- | Lazily execute a function or load from a checkpoint.
-- filename : string
-- f : function
-- The remaining arguments are to be passed into `f` if filename does not exist.
local function lazyMake(filename, f, ...)
    local result
    if paths.filep(filename) then
        result = torch.load(filename)
    else
        result = f(...)
        torch.save(filename, result)
    end
    return result
end

-- | Load data from a directory.
local function getData(path)
    assert(type(path) == 'string', 'A destination filepath must be provided.')

    os.execute('scripts/data.sh ' .. path)

    -- Load a previously generated `sst1.th7` if it exists, otherwise create
    -- a new one.
    local dic = lazyMake(
        path .. "/sst1.th7",
        data.makedictionary,
        path .. "/train.sens"
    )

    -- A smaller training corpus exists in small-train.
    local trainfilename = path .. "/train."
    local validfilename = path .. "/valid."
    local testfilename  = path .. "/test."

    local batches = {
        train = trainfilename,
        valid = validfilename,
        test  = testfilename,
    }

    for split, filename in pairs(batches) do
        -- Load sentences and convert to tensors.
        local examples = data.loadFile{
            filename      = filename .. "sens",
            dic           = dic,
            preservelines = true,
        }

        -- Load labels from the respective file.
        local labels = getLabels(filename .. 'lbls')

        -- Combine the sentence and targets into an example.
        for i, input in ipairs(examples) do
            examples[i] = {
                input  = input:view(-1, 1),
                target = labels[i],
            }
        end

        batches[split] = examples
    end

    collectgarbage()
    collectgarbage()

    return batches, dic
end

return getData
