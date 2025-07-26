# Integral Model

An integration question can be simply created using
```@repl integralModel
using SymbolicModelsUtils # hide
using Symbolics # hide
@variables x # hide

que = ∫(sin(x), ∂(x))
```

Here, `∫` is a symbolic function which takes `expression` (in this case `sin(x)`) which we wants to integrate and with respect to what (here, `∂` is a sign of differentiation)

`∂` is just a closure function of [Symbolics.Differential](https://docs.sciml.ai/Symbolics/stable/manual/derivatives/#Symbolics.Differential). To compute `∂` of any function with respect to `x` call method [apply_∂](@ref).
```@repl integralModel
apply_∂(Symbolics.value(∂(sin(x))))
```