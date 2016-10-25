-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local rnnlib = require 'rnnlib.env'

require 'nn'
require 'nngraph'

pcall(require, 'cutorch')
pcall(require, 'cunn')
pcall(require, 'cudnn')

if cudnn then
    require 'rnnlib.nn.WrappedCudnnRnn'
end

require 'rnnlib.nn.SequenceTable'
require 'rnnlib.nn.RecurrentTable'

require 'rnnlib.cell'
require 'rnnlib.make'
require 'rnnlib.recurrentnetwork'
require 'rnnlib.bidirectional'

require 'rnnlib.nn.LSTM'
require 'rnnlib.nn.RNN'
require 'rnnlib.nn.GRU'

require 'rnnlib.nn.ReverseTable'
require 'rnnlib.nn.ZipTable'

return rnnlib
