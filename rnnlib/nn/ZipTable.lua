-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- nn.ZipTable
--------------------------------------------------------------------------------

local ZipTable, parent = torch.class('nn.ZipTable', 'nn.Module')

function ZipTable:__init()
    parent.__init(self)
    self.output = {}
    self.gradInput = {}
end

function ZipTable:updateOutput(input)
    local len = #input[1]
    for i = 2, #input do
        assert(#input[i] == len, 'tables must be of equal length')
    end

    self.output = {}
    for i = 1, len do
        self.output[i] = {}
        for j = 1, #input do
            self.output[i][j] = input[j][i]
        end
    end
    return self.output
end

function ZipTable:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for i = 1, #input do
        self.gradInput[i] = {}
        for j = 1, #input[i] do
            self.gradInput[i][j] = gradOutput[j][i]
        end
    end
    return self.gradInput
end
