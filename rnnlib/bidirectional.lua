-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Utilities for Constructing Bidirectional RNNs
--------------------------------------------------------------------------------

local argcheck = require 'argcheck'
local rnnlib   = require 'rnnlib.env'
local mutils   = require 'rnnlib.mutils'

-- | Construct a bidirectional layer where the hidden states are aligned.
function rnnlib.makeBidirectionalLayer(layer, revlayer)
    local input          = nn.Identity()()

    -- Separate the forward and reverse initial hidden states
    -- and process the input sequence.
    local hids   , fwdseq = input:split(2)
    local fwdhid , revhid = hids :split(2)
    local revseq          = nn.ReverseTable()(fwdseq)

    -- Performs the computation synchronously for both directions.
    local fwdouthid , fwdoutseq = layer   { fwdhid, fwdseq }:split(2)
    local revouthid , revoutseq = revlayer{ revhid, revseq }:split(2)

    -- Reverse the output of the reverse RNN.
    revoutseq = nn.ReverseTable()(revoutseq)

    -- Combine the output.
    local outprezip = nn.ZipTable(){ fwdoutseq, revoutseq }
    local output    = nn.MapTable(nn.JoinTable(2))(outprezip)

    return nn.gModule(
        { input },
        { nn.Identity(){ fwdouthid, revouthid }, output }
    )
end

-- | Construct a multi-level bidirectional network.
rnnlib.makeBidirectional = argcheck{
    { name = 'cellfn'      , type = 'function'                   },
    { name = 'inputsize'   , type = 'number'                     },
    { name = 'hids'        , type = 'table'                      },
    { name = 'sharefields' , type = 'table'    , opt     = true  },
    { name = 'savehidden'  , type = 'boolean'  , default = false },
    call = function(cellfn, inputsize, hids, sharefields, savehidden)
        hids[0] = inputsize
        local initfs = {}
        local birnn = nn.SequenceTable(1)
        local makeLayer = rnnlib.makeBidirectionalLayer
        for i = 1, #hids do
            local c, f = cellfn(
                i == 1 and hids[i-1] or 2 * hids[i-1],
                hids[i]
            )
            initfs[i] = f
            local layer = nn.RecurrentTable{
                dim = 2,
                module = rnnlib.cell.gModule(c)
            }
            local revlayer = layer:clone(sharefields)
            birnn:add(makeLayer(layer, revlayer))
        end

        birnn.initializeHidden = function(self, bsz)
            assert(bsz and type(bsz) == 'number')
            local h = self.hiddenbuffer or {}
            for i = 1, #initfs do
                h[i] = {
                    -- Forward hidden.
                    initfs[i](bsz, self:type(), h[i] and h[i][1]),
                    -- Reverse hidden.
                    initfs[i](bsz, self:type(), h[i] and h[i][2]),
                }
            end
            self.hiddenbuffer = h
            return h
        end

        birnn.saveHidden     = savehidden
        birnn.saveLastHidden = mutils.makeSaveLastHidden()
        birnn.getLastHidden  = mutils.makeGetLastHidden()

        -- Save the hidden state when performing evaluation.
        local uo = birnn.updateOutput
        function birnn.updateOutput(self, ...)
            local output = uo(self, ...)
            if not self.train and self.saveHidden then
                self:saveLastHidden()
            end
            return output
        end

        -- Save the hidden state when accumulating the gradients wrt the
        -- parameters.
        local agp = birnn.accGradParameters
        function birnn.accGradParameters(self, ...)
            agp(self, ...)
            if self.train and self.saveHidden then
                self:saveLastHidden()
            end
        end

        -- SequenceTable has a separate backward that does not call
        -- updateGradInput and accGradParameters.
        local b = birnn.backward
        function birnn.backward(self, ...)
            local gradInput = b(self, ...)
            if self.train and self.saveHidden then
                self:saveLastHidden()
            end
            return gradInput
        end

        return birnn
    end,
}
