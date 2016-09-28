-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- Recurrent Network Convenience Function
--   Constructs the outer recurrent layer.
--------------------------------------------------------------------------------

local argcheck = require 'argcheck'
local doc      = require 'argcheck.doc'
local rnnlib   = require 'rnnlib.env'

doc[[

### RNN Make

This function provides an interface for constructing the outer layer of a
two-dimensional recurrent network, where the first dimension refers to
the hidden layer depth and the second refers to time.

When given a SequenceTable, we assume the outer dimension should have
weight sharing. When given a RecurrentTable, we assume the outer dimension
should not.

]]

local make = argcheck{
    { name = "outerdim" , type = "number"   ,            },
    { name = "model"    , type = "table"    ,            },
    { name = "hinitfn"  , type = "function" , opt = true },
    call = function(outerdim, model, hinitfn)
        return nn.SequenceTable(outerdim, model, hinitfn)
    end
}

make = argcheck{
    { name = "outerdim" , type = "number"    ,            },
    { name = "model"    , type = "nn.Module" ,            },
    { name = "hinitfn"  , type = "function"  , opt = true },
    overload = make,
    call = function(outerdim, model, hinitfn)
        return nn.RecurrentTable(outerdim, model, hinitfn)
    end
}

make = argcheck{
    { name = "outerdim" , type = "number"   ,            },
    { name = "model"    , type = "function" ,            },
    { name = "hinitfn"  , type = "function" , opt = true },
    overload = make,
    call = function(outerdim, model, hinitfn)
        return nn.RecurrentTable(outerdim, rnnlib.cell.gModule(model), hinitfn)
    end
}

rnnlib.make = make
