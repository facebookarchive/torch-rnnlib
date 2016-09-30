# Modules

## nn.RNN

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

## nn.LSTM

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

## nn.GRU

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

## nn.SequenceTable

```lua
nn.SequenceTable{
    dim     = number,
   [modules = table],
   [initfun = function],
}
```

This container applies a sequence of modules to an input sequence along a given
dimension. Each module in self.modules takes as input both the state of the
previous module and the next item in the input sequence. The output of the
internal modules should be of the same dimension as the initial input.

The container accumulates the outputs of the modules and returns them as a
table.

See `rnnlib/nn/SequenceTable` for more details.

## nn.RecurrentTable

```lua
nn.RecurrentTable{
    dim         = number,
    module      = nn.Module,
   [initfun     = function],
   [sharedfield = table],
}
```

This module is a wrapper around nn.SequenceTable where the same module is cloned in order to match the length of the input sequence.