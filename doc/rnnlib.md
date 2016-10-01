# rnnlib

## [rnnlib.setupRecurrent](../rnnlib/recurrentnetwork.lua#L16)

```lua
rnnlib.setupRecurrent{
    network    = nn.SequenceTable,
    initfs     = table,
   [hinitfun   = function],
   [winitfun   = function (mutils.defwinitfun)],
   [savehidden = boolean (false)],
}
```

Given a recurrent network `network` and a table of initialization functions `initfs`
this function adds member functions to `network` that facilitate initializing and
saving the hidden state at each timestep. You should use `network.hiddenbuffer`
as the hidden input, and always be sure to call `network:initializeHidden(bsz)`
before performing any computation.

## [rnnlib.makeRecurrent](../rnnlib/recurrentnetwork.lua#L87)

```lua
rnnlib.makeRecurrent{
    cellfn     = function,
    inputsize  = number,
    hids       = table,
   [hinitfun   = function],
   [winitfun   = function (mutils.defwinitfun)],
   [savehidden = boolean (true)],
}
```

Given a cell function, e.g. from [rnnlib.cell](cell.md), `makeRecurrent` constructs an
RNN whose outer layer is of type nn.SequenceTable and inner layers are of type
nn.RecurrentTable.

Be sure to call :training() or :evaluate() before performing training or
evaluation, since the hidden state logic depends on this. If you would rather
handle this manually or do not need to use the hidden state in the next batch,
set .saveHidden = false.

## [rnnlib.makeCudnnRecurrent](../rnnlib/recurrentnetwork.lua#L143)

```lua
rnnlib.makeRecurrent{
    cellstring = string,
    inputsize  = number,
    hids       = table,
   [hinitfun   = function],
   [winitfun   = function (mutils.defwinitfun)],
   [savehidden = boolean (true)],
}
```

Create a recurrent network with the same API as the above, but with
[torch Cudnn bindings](https://github.com/soumith/cudnn.torch) as a backend.

## [rnnlib.makeBidirectional](../rnnlib/bidirectional.lua#L44)

```lua
rnnlib.makeBidirectional{
    cellfn      = function,
    inputsize   = number,
    hids        = table,
   [sharefields = table],
   [savehidden  = boolean (true)],
}
```

Create a bidirectional recurrent neural network.