-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- A constructor for creating homogeneous GRU networks.
--------------------------------------------------------------------------------

require 'nn'

local argcheck = require 'argcheck'
local rnnlib   = require 'rnnlib.env'

nn.GRU = argcheck{
    { name = "inputsize" , type = "number"   ,                 },
    { name = "hidsize"   , type = "number"   ,                 },
    { name = "nlayer"    , type = "number"   ,                 },
    { name = "hinitfun"  , type = "function" , opt = true      },
    { name = "winitfun"  , type = "function" , opt = true      },
    { name = 'usecudnn'  , type = 'boolean'  , default = false },
    call = function(inputsize, hidsize, nlayer, hinitfun, winitfun, usecudnn)
        local hids = {}
        for i = 1, nlayer do
            hids[i] = hidsize
        end
        local model = rnnlib.makeRecurrent{
            cellfn    = rnnlib.cell.GRU,
            inputsize = inputsize,
            hids      = hids,
            hinitfun  = hinitfun,
            winitfun  = winitfun,
        }
        if usecudnn then
            return nn.WrappedCudnnRnn(model, 'GRU', hids, model.saveHidden)
        end
        return model
    end
}
