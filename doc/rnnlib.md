# rnnlib

## rnnlib.setupRecurrent

```lua
rnnlib.setupRecurrent{
    network    = nn.SequenceTable,
    initfs     = table,
   [hinitfun   = function],
   [winitfun   = function (mutils.defwinitfun)],
   [savehidden = boolean (false)],
}
```

## rnnlib.makeRecurrent

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

## rnnlib.makeCudnnRecurrent

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

## rnnlib.makeBidirectional

```lua
rnnlib.makeBidirectional{
    cellfn      = function,
    inputsize   = number,
    hids        = table,
   [sharefields = table],
   [savehidden  = boolean (true)],
}
```