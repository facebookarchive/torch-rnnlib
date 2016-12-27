# rnnlib.cell

A collection of utilities for working with and implementations of cells for recurrent networks.
All cells consist of a pair of functions, `make` and `init`.
`make` takes in the previous hidden state and input and returns the following state and cell output.
Generally this will then be fed into `cell.gModule` to create an `nn.gModule`.
`init` initializes the hidden state of a cell. The default for this is `:fill(0)`.

## [cell.flip](../rnnlib/cell.lua#L54)

```lua
cell.flip(function)
```

Takes a function (a,b) -> (a,b) and returns a function (b,a) -> (b,a).
You could use this to change the cell functions from
taking arguments (state, input) to (input, state).

## [cell.gModule](../rnnlib/cell.lua#L62)

```lua
cell.gModule(function)
```

Create a gModule from a cell function. Used to construct an nn.Module
from a cell's `make` function before being passed into an
(nn.RecurrentTable)[modules.md#nnrecurrenttable]'s constructor.

## [cell.wrapOutput](../rnnlib/cell.lua#L69)

```lua
cell.wrapOutput(make, constructor, ...)
```

Wraps the output of a cell with another module.
Recall that a cell's output looks like (state, output).
The constructor should be an nn.Module constructor such as nn.Identity or nn.Dropout.
The variadic arguments at the end are fed to the constructor to produce an nn.Module.

## [cell.LSTM](../rnnlib/cell.lua#L108)

```lua
cell.LSTM(nin, nhid)
```

Constructs an LSTM cell.
`nin` is the size of the input dimension and `nhid` is the size of the hidden dimension.

## [cell.RNNSigmoid](../rnnlib/cell.lua#L96)

```lua
cell.RNNSigmoid(nin, nhid)
```

Constructs an Elman RNN cell with sigmoid nonlinearities.
`nin` is the size of the input dimension and `nhid` is the size of the hidden dimension.

## [cell.RNNTanh](../rnnlib/cell.lua#L100)

```lua
cell.RNNTanh(nin, nhid)
```

Constructs an RNN cell with tanh nonlinearities.
`nin` is the size of the input dimension and `nhid` is the size of the hidden dimension.

## [cell.RNNReLU](../rnnlib/cell.lua#L104)

```lua
cell.RNNReLU(nin, nhid)
```

Constructs an RNN cell with ReLU nonlinearities.
`nin` is the size of the input dimension and `nhid` is the size of the hidden dimension.

## [cell.GRU](../rnnlib/cell.lua#L150)

```lua
cell.GRU(nin, nhid)
```

Constructs a GRU cell.
`nin` is the size of the input dimension and `nhid` is the size of the hidden dimension.
