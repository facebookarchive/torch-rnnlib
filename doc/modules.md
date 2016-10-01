# Modules

All modules in this file take as input
```lua
let
  input    = { hidden, sequence }
where
  hidden   = { state(bsz) }       -- a table of states corresponding to the initial states of each layer
  sequence = { bsz x inputDim }   -- a table of input tensors of dimension bsz x inputDim
```
Here's an example of an n-layered LSTM network:
```lua
let
  input  = { hidden, sequence }
where
  hidden = {
    {
      torch.Tensor(bsz, hiddenDim), -- This is the cell state.
      torch.Tensor(bsz, hiddenDim), -- This is the hidden state.
    }, -- This is a single cell's state.
    -- ... You would repeat this n-1 more times.
  }
  sequence = {
    torch.Tensor(bsz, inputDim),
    -- ... You would repeat this for bptt-1 time steps.
  }
```
For an n-layer RNN network, you would simply change
```lua
  hidden = {
    torch.Tensor(bsz, hiddenDim) -- This is a single cell's state.
    -- ... Again, you would just repeat this n-1 more times.
  }
```

## [nn.RNN](../rnnlib/nn/RNN.lua)

```lua
nn.RNN{
    inputsize = number,
    hidsize   = number,
    nlayer    = number,
   [hinitfun  = function],
   [winitfun  = function],
   [usecudnn  = boolean (false)],
}
```

Constructs an RNN network using the [RNN](cell.md#cellelman) cell.

## [nn.LSTM](../rnnlib/nn/LSTM.lua)

```lua
nn.LSTM{
    inputsize = number,
    hidsize   = number,
    nlayer    = number,
   [hinitfun  = function],
   [winitfun  = function],
   [usecudnn  = boolean (false)],
}
```

Constructs an LSTM network using the [LSTM](cell.md#celllstm) cell.

## [nn.GRU](../rnnlib/nn/GRU.lua)

```lua
nn.GRU{
    inputsize = number,
    hidsize   = number,
    nlayer    = number,
   [hinitfun  = function],
   [winitfun  = function],
   [usecudnn  = boolean (false)],
}
```

Constructs an LSTM network using the [GRU](cell.md#cellgru) cell.

## [nn.SequenceTable](../rnnlib/nn/SequenceTable.lua)

```lua
nn.SequenceTable{
    dim     = number,
   [modules = table],
   [initfun = function],
}
```

This is a subclass of [nn.Container](https://github.com/torch/nn/blob/master/Container.lua)
that applies a sequence of modules to an input sequence along a given
dimension. Each module in self.modules takes as input both the state of the
previous module and the next item in the input sequence. The output of the
internal modules should be of the same dimension as the initial input.

The container accumulates the outputs of the modules and returns them as a
table.

See [SequenceTable.lua](../rnnlib/nn/SequenceTable.lua) for more details.

## [nn.RecurrentTable](../rnnlib/nn/RecurrentTable.lua)

```lua
nn.RecurrentTable{
    dim         = number,
    module      = nn.Module,
   [initfun     = function],
   [sharedfield = table],
}
```

This module is a wrapper around [nn.SequenceTable](../rnnlib/nn/SequenceTable.lua)
where the same module is cloned in order to match the length of the input sequence.