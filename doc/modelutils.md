# Model Utilities

## [utils.inmodule](../rnnlib/mutils.lua#L170)

```lua
utils.inmodule(model, inmodule)
```

Given a recurrent network `model` and another
[nn.Module](https://github.com/torch/nn/blob/master/Module.lua)
`inmodule` that will be applied at each timestep.
The second index of the input is assumed to be the input over time.
Note that this assumes that the input will be a *table* of length `time`,
or more specifically that the output of `inmodule` will be of dimension
`bsz x dimension`.

## [utils.batchedinmodule](../rnnlib/mutils.lua#L182)

```lua
utils.batchedinmodule(model, inmodule)
```

Given a recurrent network `model` and another
[nn.Module](https://github.com/torch/nn/blob/master/Module.lua)
`inmodule` that will be applied at each timestep.
The second index of the input is assumed to be the input over time.
Note that this assumes that the input will be a *tensor* of length `time`,
or more specifically that the output of `inmodule` will be of dimension
`seqlen x bsz x dimension`.

## [utils.outmodule](../rnnlib/mutils.lua#L195)

```lua
utils.outmodule(model, outmodule)
```

Given a recurrent network `model` and another
[nn.Module](https://github.com/torch/nn/blob/master/Module.lua)
`outmodule` will be mapped over the top-level table of the
output of the network.