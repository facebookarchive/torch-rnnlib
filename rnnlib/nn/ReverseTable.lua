-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- nn.ReverseTable
--------------------------------------------------------------------------------

local ReverseTable, parent = torch.class('nn.ReverseTable', 'nn.Module')

function ReverseTable:__init()
    parent.__init(self)
end

function ReverseTable:updateOutput(input)
    self.output = {}
    local len = #input
    for i = 1, len do
        self.output[i] = input[len-i+1]
    end
    return self.output
end

function ReverseTable:updateGradInput(input, gradOutput)
    self.gradInput = {}
    local len = #gradOutput
    for i = 1, len do
        self.gradInput[i] = gradOutput[len-i+1]
    end
    return self.gradInput
end
