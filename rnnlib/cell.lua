-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Cells for RNNs
--------------------------------------------------------------------------------

local doc    = require 'argcheck.doc'
local rnnlib = require 'rnnlib.env'

doc[[

### RNN Cells

This file contains implementations of Elman RNN, LSTM, and GRU cells for
constructing full recurrent networks with SequenceTables and RecurrentTables.

The cell constructors return pairs of `make` and `init` functions. The `make`
function can be fed into `cell.gModule` to construct a gModule, while the
`init` function can be used to make a closure that initializes the hidden
state of the network that contains the cell.

Once turned into a gModule, all cells take as input a tuple of (state, input)
and return a tuple of (state, output). This can be reversed with `cell.flip`
if one changes the ordering of the dimensions.

Here's an example that creates a single-layer LSTM network:

```
local cellfun, initfun = cell.LSTM(5, 10)
local gmodule = cell.gModule(c)
local network = nn.RecurrentTable{ dim = 2, module = gmodule }

network.initializeHidden = function(self, bsz) return { initfun(bsz) } end
```

]]

local cell = {}

-- | A basic initialization function for hidden states.
-- Reuses memory if available.
local _init = function(bsz, nhid, t, cache)
    local t = t or 'torch.Tensor'
    local tensor = cache or torch.Tensor()
    return tensor:resize(bsz, nhid):type(t):fill(0)
end

-- | Takes a function (a,b) -> (a,b) and returns a function (b,a) -> (b,a).
cell.flip = function(f)
    return function(a, b)
        local x, y = f(b, a)
        return y, x
    end
end

-- | Create a gModule from a cell function.
cell.gModule = function(fn)
    local input = nn.Identity()()
    local state = nn.Identity()()
    return nn.gModule({state, input}, {fn(state, input)})
end

-- | Wrap the output of a cell with another module.
cell.wrapOutput = function(make, constructor, ...)
    local module = constructor(...)
    return function(...)
        local nexth, output = make(...)
        return nexth, module(output)
    end
end

--------------------------------------------------------------------------------
-- Cell definitions
--------------------------------------------------------------------------------

local function makeRnn(nin, nhid, nonlinearity)
    local make = function(prevh, input)
        local i2h   = nn.Linear(nin,  nhid, false)(input):annotate{name="rnn_i2h"}
        local h2h   = nn.Linear(nhid, nhid, false)(prevh):annotate{name="rnn_h2h"}
        local nexth = nonlinearity()(nn.CAddTable()({i2h, h2h})):annotate{name="rnn_nexth"}
        return nexth, nn.Identity()(nexth)
    end

    local init = function(bsz, t, cache)
        return _init(bsz, nhid, t, cache)
    end

    return make, init
end

cell.Elman = function(nin, nhid)
    return makeRnn(nin, nhid, nn.Sigmoid)
end

cell.RNNTanh = function(nin, nhid)
    return makeRnn(nin, nhid, nn.Tanh)
end

cell.RNNReLU = function(nin, nhid)
    return makeRnn(nin, nhid, nn.ReLU)
end

cell.LSTM = function(nin, nhid)
    local make = function(prevch, input)
        -- prevch : { prevc : node, prevh : node }
        -- input : node
        local split = {prevch:split(2)}
        local prevc = split[1]
        local prevh = split[2]

        -- the four gates are computed simulatenously
        local i2h   = nn.Linear(nin,  4 * nhid, false)(input):annotate{name="lstm_i2h"}
        local h2h   = nn.Linear(nhid, 4 * nhid, false)(prevh):annotate{name="lstm_h2h"}
        -- the gates are separated
        local gates = nn.CAddTable()({i2h, h2h})
        -- assumes that input is of dimension nbatch x ngate * nhid
        gates = nn.SplitTable(2)(nn.Reshape(4, nhid)(gates))
        -- apply nonlinearities:
        local igate = nn.Sigmoid()(nn.SelectTable(1)(gates)):annotate{name="lstm_ig"}
        local fgate = nn.Sigmoid()(nn.SelectTable(2)(gates)):annotate{name="lstm_fg"}
        local cgate = nn.Tanh   ()(nn.SelectTable(3)(gates)):annotate{name="lstm_cg"}
        local ogate = nn.Sigmoid()(nn.SelectTable(4)(gates)):annotate{name="lstm_og"}
        -- c_{t+1} = fgate * c_t + igate * f(h_{t+1}, i_{t+1})
        local nextc = nn.CAddTable()({
            nn.CMulTable()({fgate, prevc}),
            nn.CMulTable()({igate, cgate})
        }):annotate{name="nextc"}
        -- h_{t+1} = ogate * c_{t+1}
        local nexth  = nn.CMulTable()({ogate, nn.Tanh()(nextc)}):annotate{name="lstm_nexth"}
        local nextch = nn.Identity ()({nextc, nexth}):annotate{name="lstm_nextch"}
        return nextch, nexth
    end

    -- initialize function:
    local init = function(bsz, t, cache)
        return {
            _init(bsz, nhid, t, cache and cache[1]),
            _init(bsz, nhid, t, cache and cache[2])
        }
    end

    return make, init
end

cell.GRU = function(nin, nhid)
    local make = function(prevh, input)
        -- [ W_reset x, W_update x, W x ]
        local i2h = nn.Linear(nin,  3 * nhid, false)(input):annotate{name="gru_i2h"}
        -- [ U_reset prevh, U_update prevh, U prevh ]
        local h2h = nn.Linear(nhid, 3 * nhid, false)(prevh):annotate{name="gru_h2h"}
        -- [ W_reset x + U_reset prevh, W_update x + U_update prevh ]
        local ru = nn.CAddTable()({
            nn.Narrow(2, 1, 2*nhid)(i2h),
            nn.Narrow(2, 1, 2*nhid)(h2h),
        })
        -- Assumes that input is of dimension nbatch x ngate * nhid.
        ru = nn.SplitTable(2)(nn.Reshape(2, nhid)(ru)):annotate{name="gru_ru"}
        local rgate = nn.Sigmoid()(nn.SelectTable(1)(ru)):annotate{name="gru_rgate"}
        local ugate = nn.Sigmoid()(nn.SelectTable(2)(ru)):annotate{name="gru_ugate"}
        -- Use formulation in Cho et al 2014:
        -- output = tanh(W x + rgate .* U prevh).
        local output = nn.Tanh()(nn.CAddTable()({
            nn.Narrow(2, 2*nhid+1, nhid)(i2h),
            nn.CMulTable()({
                rgate,
                nn.Narrow(2, 2*nhid+1, nhid)(h2h),
            })
        })):annotate{name="gru_output"}
        -- nexth = (1-ugate) output + ugate prevh
        local nexth = nn.CAddTable()({
            output,
            nn.CMulTable()({
                ugate,
                nn.CSubTable()({
                    prevh,
                    output,
                }),
            }),
        }):annotate{name="gru_nexth"}
        return nexth, nn.Identity()(nexth)
    end

    local init = function(bsz, t, cache)
        return _init(bsz, nhid, t, cache)
    end

    return make, init
end

rnnlib.cell = cell

return cell
