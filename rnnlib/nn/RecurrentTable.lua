-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

--------------------------------------------------------------------------------
-- RecurrentTable
--   This is a subclass of SequenceTable which takes care of the cloning of the
-- recurrent module.
--------------------------------------------------------------------------------

local argcheck = require 'argcheck'

local RecurrentTable, parent = torch.class('nn.RecurrentTable', 'nn.SequenceTable')

-- | Initialize the model.
-- `sharedfields` gives a table that must be unpacked for weight sharing.
RecurrentTable.__init = argcheck{
    doc = [[
<a name="RecurrentTable">
#### nn.RecurrentTable(@ARGP)
@ARGT

    nn.RecurrentTable{
        dim: number, module: nn.Module, initfun: Opt function, sharedfields: Opt [fields]
    }

This module is a wrapper around `nn.SequenceTable` where the same module is
cloned in order to match the length of the input sequence.
]],
    { name = 'self'         , type = 'nn.RecurrentTable' ,            },
    { name = 'dim'          , type = 'number'            ,            },
    { name = 'module'       , type = 'nn.Module'         ,            },
    { name = 'initfun'      , type = 'function'          , opt = true },
    { name = 'sharedfields' , type = 'table'             , opt = true },
    call = function(self, dim, module, initfun, sharedfields)
        parent.__init(self, dim, {module}, initfun)
        self.sharedfields  = sharedfields or {
            'weight' , 'gradWeight' ,
            'bias'   , 'gradBias'   ,
        }
    end
}

RecurrentTable.add = function(...) error('Cannot add module to RecurrentTable') end

-- | Unroll the core model (the column model)
-- to create an RNN. We unroll until self.modules is of length `size`.
-- This is lazy and does *not* reclone.
RecurrentTable.extend = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    { name = 'size' , type = 'number'            },
    call =  function(self, size)
        assert(self.modules and #self.modules >= 1)
        assert(self.sharedfields)
        for t = #self.modules + 1, size do
            self.modules[t] = self.modules[1]:clone(table.unpack(self.sharedfields))
        end
    end
}

-- Forcibly resize an RNN to `size` timesteps.
-- This deletes all the previous clones.
-- Please invoke the garbage collector after calling this method.
RecurrentTable.resize = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    { name = 'size' , type = 'number'            },
    call = function(self, size)
        assert(self.modules and #self.modules >= 1)
        assert(self.sharedfields)
        for t = #self.modules, 2, -1 do
            self.modules[t] = nil
        end
        self:extend(size)
    end
}

-- | Apply a function to the main module.
-- This does not overload the :applyToModules() function so stuff like
-- :training() and :evaluate() are not changed.
RecurrentTable.apply = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    { name = 'fun'  , type = 'function'          },
    call = function(self, fun)
        assert(self.modules and #self.modules >= 1)
        -- The function is only applied to the original module.
        -- This prevents strange things from happening with functions that are
        -- not idempotent.
        self.modules[1]:apply(fun)
    end
}

-- | Return the parameters from the only module in the container.
RecurrentTable.parameters = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    call = function(self)
        assert(self.modules and #self.modules >= 1)
        return self.modules[1]:parameters()
    end
}

-- | :getParameters() makes all clones point to an invalid region in memory,
-- so all clones *must be re-cloned*.
RecurrentTable.getParameters = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    call = function(self)
        local w, dw = self.modules[1]:getParameters()
        -- Recall the resize removes all clones and reclones.
        self:resize(#self.modules)
        return w, dw
    end
}

RecurrentTable.updateParameters = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    { name = 'lr'   , type = 'number'            },
    call = function(self, lr)
        assert(self.modules and #self.modules >= 1)
        assert(lr >= 0)
        self.modules[1]:updateParameters(lr)
    end
}

RecurrentTable.zeroGradParameters = argcheck{
    { name = 'self' , type = 'nn.RecurrentTable' },
    call = function (self)
        assert(self.modules and #self.modules >= 1)
        self.modules[1]:zeroGradParameters()
    end
}

-- | Replicates the core module until it is long enough and then performs the
-- forward propagation using SequenceTable's updateOutput.
RecurrentTable.updateOutput = argcheck{
    { name = 'self'  , type = 'nn.RecurrentTable' },
    { name = 'input' , type = 'table'             },
    call = function(self, input)
        -- wrap singleton dimensions
        if torch.isTensor(input[self.dim]) then
            input[self.dim] = {input[self.dim]}
        end
        self:extend(#input[self.dim])
        return parent.updateOutput(self, input)
    end
}
