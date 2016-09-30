-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- SequenceTable
--------------------------------------------------------------------------------

require 'nngraph'
require 'cutorch'

local argcheck = require 'argcheck'
local mutils   = require 'rnnlib.mutils'

local SequenceTable, parent = torch.class('nn.SequenceTable', 'nn.Container')

-- | Perform a shallow copy where only the table structure is replicated.
local function _shallowcopy(tbl)
    local ntbl = {}
    for k,v in pairs(tbl) do
        ntbl[k] = v
    end
    return ntbl
end

-- | Initialize the model.
SequenceTable.__init = argcheck{
    doc = [[
<a name="SequenceTable">
#### nn.SequenceTable(@ARGP)
@ARGT

This container applies a sequence of modules to an input sequence along a given
dimension. Each module in self.modules takes as input both the state of the
previous module and the next item in the input sequence. The output of the
internal modules should be of the same dimension as the initial input.

The container accumulates the outputs of the modules and returns them as a
table.

More specifically, the input is a table which contains a sequence in at least
one of its elements. The SequenceTable must be initialized with a specific
index of the input table to iterate over (the aforementioned index in the table
is referred to as `dim`).

One could view this container as an augmentation of nn.Sequential which
takes an additional input stored in the `dim` dimension of the input table.

Some preliminaries: foldl is a reduction over a list from left to right
where a function is applied to each element and an accumulated value.

The forward operation for SequenceTable can then be expressed loosely in
pseudocode as

```
tail $ scanl
    (\state (f, in) -> f state in)
    input[forall d ~= dim]
    (zip self.modules input[dim])
```

so that the output is actually the history of partial computation at each
step in the unrolled network, whereas nn.Sequential would be

```
 foldl
     (\state f -> f state)
     input
     self.modules
```

since only the final result is returned and there is no external input at each
point in the unrolled network.

Here's an example with the recurrent function `f := \ a x -> ax + a`,
where `a` is the previous hidden state and `x` is the input:

Let

```
input = {
    torch.Tensor{ 2 },
    torch.Tensor{ 1, 2, 3, 4 }:split(1),
}
```

One can then construct a single-layer recurrent network with

```
rnnlib = require 'rnnlib'
f = function(hid, input)
    local ax   = nn.CMulTable(){hid, input}
    local axpa = nn.CAddTable(){ax, hid}
    return axpa, nn.Identity()(axpa)
end
c = rnnlib.cell.gModule(f)
network = nn.RecurrentTable{
    dim    = 2,
    module = c,
}
```

The output of `network:forward(input)` will give something like

```
{
    torch.Tensor{ 4, 12, 48, 240 }:split(1),
    torch.Tensor{ 4, 12, 48, 240 }:split(1),
}
```

since the recurrent cell returns the same result for both the hidden state
and module output.

For a more detailed explanation and example (multi-layered) see
 `rnnlib/test_sequence.lua` and `examples/word-language-model/word_lm.lua`.
]],
    { name = 'self'    , type = 'nn.SequenceTable' ,            },
    { name = 'dim'     , type = 'number'           ,            },
    { name = 'modules' , type = 'table'            , opt = true },
    { name = 'initfun' , type = 'function'         , opt = true },
    call = function(self, dim, modules, initfun)
        parent.__init(self)
        -- This allows the user to manipulate the input during a forward pass.
        self.initfun = initfun
        self.modules = modules or self.modules
        self.dim     = dim
    end
}

-- | Perform a forward propagation through the SequenceTable.
-- Acts in a similar fashion as nn.Sequential but with an additional input and
-- output for each module (stored along the dim-th dimension).
-- input: a table such that `input[dim]` is a sequence.
-- output: a table of the size #input where output[dim] is a sequence of output.
SequenceTable.updateOutput = argcheck{
    { name = 'self'  , type = 'nn.SequenceTable' },
    { name = 'input' , type = 'table'            },
    call = function(self, input)
        local d = self.dim
        if self.initfun then
            input = self.initfun(input, self.inputs[#self.inputs], d)
        end
        self.inputs = {}
        self.output = {}
        local currentOutput = _shallowcopy(input)
        for t = 1, #input[d] do
            currentOutput[d] = input[d][t]
            self.inputs[t]   = currentOutput
            currentOutput    = self.modules[t]:forward(currentOutput)
            self.output[t]   = _shallowcopy(currentOutput)
        end
        self.output = mutils.transpose(self.output)
        return self.output
    end
}

-- | Calculates the gradient with respect to the model's input.
-- Accumulates the gradients along `self.dim` and returns only the
-- final gradients for all dimension not equal to `self.dim`.
-- Again, in that respect this is very similar to a nn.Sequential's
-- backward, except that along `self.dim` the gradients are
-- accumulated into a list.
SequenceTable.updateGradInput = argcheck{
    { name = 'self'       , type = 'nn.SequenceTable' },
    { name = 'input'      , type = 'table'            },
    { name = 'gradOutput' , type = 'table'            },
    call = function(self, input, gradOutput)
        local d    = self.dim
        -- `transpose` creates a shallowcopy, so the gradOutput is not mutated.
        gradOutput = mutils.transpose(gradOutput)
        local currentGradOutput = gradOutput[#input[d]]
        local gradInputAlongD = {}
        for t = #input[d], 1, -1 do
            currentGradOutput[d] = gradOutput[t][d]
            currentGradOutput    = _shallowcopy(
                self.modules[t]:updateGradInput(
                    self.inputs[t], currentGradOutput)
            )
            gradInputAlongD[t] = currentGradOutput[d]
        end
        currentGradOutput[d] = gradInputAlongD
        self.gradInput = currentGradOutput
        return self.gradInput
    end
}

-- | Accumulates the gradient wrt the parameters.
-- Uses the gradInputs accumulated in :updateGradInput.
SequenceTable.accGradParameters = argcheck{
    { name = 'self'       , type = 'nn.SequenceTable'               },
    { name = 'input'      , type = 'table'                          },
    { name = 'gradOutput' , type = 'table'                          },
    { name = 'scale'      , type = 'number'           , default = 1 },
    call = function(self, input, gradOutput, scale)
        local d        = self.dim
        -- `transpose` creates a shallowcopy, so the gradOutput is not mutated.
        gradOutput     = mutils.transpose(gradOutput)
        local currentGradOutput = gradOutput[#input[d]]
        for t = #input[d], 1, -1 do
            currentGradOutput[d] = gradOutput[t][d]
            self.modules[t]:accGradParameters(
                self.inputs[t], currentGradOutput, scale)
            currentGradOutput = _shallowcopy(self.modules[t].gradInput)
        end
    end
}

-- | The backward pass rewritten here for convenience.
SequenceTable.backward = argcheck{
    { name = 'self'       , type = 'nn.SequenceTable'               },
    { name = 'input'      , type = 'table'                          },
    { name = 'gradOutput' , type = 'table'                          },
    { name = 'scale'      , type = 'number'           , default = 1 },
    call = function(self, input, gradOutput, scale)
        local d    = self.dim
        -- `transpose` creates a shallowcopy, so the gradOutput is not mutated.
        gradOutput = mutils.transpose(gradOutput)
        local currentGradOutput = gradOutput[#input[d]]
        local gradInputAlongD = {}
        for t = #input[d], 1, -1 do
            currentGradOutput[d] = gradOutput[t][d]
            currentGradOutput    = _shallowcopy(
                self.modules[t]:backward(
                    self.inputs[t], currentGradOutput, scale)
            )
            gradInputAlongD[t] = currentGradOutput[d]
        end
        currentGradOutput[d] = gradInputAlongD
        self.gradInput = currentGradOutput
        return self.gradInput
    end
}
