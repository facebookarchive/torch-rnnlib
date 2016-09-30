# rnnlib

## [rnnlib.setupRecurrent](../rnnlib/recurrentnetwork.lua#L17)

```lua
rnnlib.setupRecurrent{
    network    = nn.SequenceTable,
    initfs     = table,
   [hinitfun   = function],
   [winitfun   = function (mutils.defwinitfun)],
   [savehidden = boolean (false)],
}
```

## [rnnlib.makeRecurrent](../rnnlib/recurrentnetwork.lua#L88)

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

## [rnnlib.makeCudnnRecurrent](../rnnlib/recurrentnetwork.lua#L144)

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