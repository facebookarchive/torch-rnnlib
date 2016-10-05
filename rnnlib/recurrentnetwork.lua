-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Recurrent Network Constructor
--------------------------------------------------------------------------------

local argcheck = require 'argcheck'
local mutils   = require 'rnnlib.mutils'
local rnnlib   = require 'rnnlib.env'

rnnlib.setupRecurrent = argcheck{
    doc = [[
<a name="setupRecurrent">
### rnnlib.setupRecurrent(@ARGP)
@ARGT

Setup the stateful part of a recurrent network.
Take note that if you have a single layer recurrent network
(nn.Recurrent Table) without an nn.SequenceTable wrapper, you may have to
index into the hiddenbuffer like so: rnn.hiddenbuffer[1] since `initfs`
is required to be a table of functions.
]],
    { name = 'network'    , type = 'nn.SequenceTable' ,                              },
    { name = 'initfs'     , type = 'table'            ,                              },
    { name = "winitfun"   , type = "function"         , default = mutils.defwinitfun },
    { name = "savehidden" , type = "boolean"          , default = false              },
    call = function(network, initfs, winitfun, savehidden)
        -- Initialize weights.
        winitfun(network)

        -- Initialize hidden states.
        network.initializeHidden = function(self, bsz)
            assert(bsz and type(bsz) == 'number')
            local h = self.hiddenbuffer or {}
            for i = 1, #initfs do
                h[i] = initfs[i](bsz, self:type(), h[i])
            end
            self.hiddenbuffer = h
            return h
        end

        network.saveHidden     = savehidden
        network.saveLastHidden = mutils.makeSaveLastHidden()
        network.getLastHidden  = mutils.makeGetLastHidden()

        -- Save the hidden state when performing evaluation.
        local uo = network.updateOutput
        function network.updateOutput(self, ...)
            local output = uo(self, ...)
            if not self.train and self.saveHidden then
                self:saveLastHidden()
            end
            return output
        end

        -- Save the hidden state when accumulating the gradients wrt the
        -- parameters.
        local agp = network.accGradParameters
        function network.accGradParameters(self, ...)
            agp(self, ...)
            if self.train and self.saveHidden then
                self:saveLastHidden()
            end
        end

        -- SequenceTable has a separate backward that does not call
        -- updateGradInput and accGradParameters.
        local b = network.backward
        function network.backward(self, ...)
            local gradInput = b(self, ...)
            if self.train and self.saveHidden then
                self:saveLastHidden()
            end
            return gradInput
        end

        return network
    end
}

rnnlib.makeRecurrent = argcheck{
        doc = [[
<a name="makeRecurrent">
### rnnlib.makeRecurrent(@ARGP)
@ARGT

Given a cell function, e.g. from 'rnnlib.cell', `makeRecurrent` constructs an
RNN whose outer layer is of type nn.SequenceTable and inner layers are of type
nn.RecurrentTable.

Be sure to call :training() or :evaluate() before performing training or
evaluation, since the hidden state logic depends on this. If you would rather
handle this manually, set .saveHidden = false.
]],
        { name = "cellfn"     , type = "function" ,                              },
        { name = "inputsize"  , type = "number"   ,                              },
        { name = "hids"       , type = "table"    ,                              },
        { name = "hinitfun"   , type = "function" , opt     = true               },
        { name = "winitfun"   , type = "function" , default = mutils.defwinitfun },
        { name = "savehidden" , type = "boolean"  , default = true               },
        call = function(cellfn, inputsize, hids, hinitfun, winitfun, savehidden)
            hids[0] = inputsize

            local initfs = {}
            local nlayer = #hids

            -- If you wanted to unroll over depth first, you could do so by:
            -- local m = nn.SequenceTable{ dim = 1 }
            -- for i = 1, nlayer do
            --     local c, f = cellfn(hids[i-1], hids[i])
            --     m:add(cell.gModule(c))
            --     initfs[i] = f
            -- end
            -- network = rnnlib.make(2, m, hinitfun)

            local layers = {}
            for i = 1, nlayer do
                local c, f = cellfn(hids[i-1], hids[i])
                layers[i] = nn.RecurrentTable{
                    dim    = 2,
                    module = rnnlib.cell.gModule(c),
                }
                initfs[i] = f
            end

            local network = rnnlib.make(1, layers, hinitfun)
            return rnnlib.setupRecurrent{
                network = network,
                initfs = initfs,
                winitfun = winitfun,
                savehidden = savehidden,
            }
        end
}

rnnlib.makeCudnnRecurrent = argcheck{
        doc = [[
<a name="makeCudnnRecurrent">
### rnnlib.makeCudnnRecurrent(@ARGP)
@ARGT

Given a cell string, `makeCudnnRecurrent` constructs a RNN using Cudnn.
]],
        { name = "cellstring" , type = "string"   ,                              },
        { name = "inputsize"  , type = "number"   ,                              },
        { name = "hids"       , type = "table"    ,                              },
        { name = "hinitfun"   , type = "function" , opt     = true               },
        { name = "winitfun"   , type = "function" , default = mutils.defwinitfun },
        { name = "savehidden" , type = "boolean"  , default = true               },
        call = function(cellstring, inputsize, hids, hinitfun, winitfun, savehidden)
            hids[0] = inputsize
            local rnn = rnnlib.makeRecurrent{
                cellfn    = rnnlib.cell[cellstring],
                inputsize = inputsize,
                hids      = hids,
                hinitfun  = hinitfun,
                winitfun  = winitfun,
            }

            return nn.WrappedCudnnRnn(rnn, cellstring, hids, savehidden)
        end
}
