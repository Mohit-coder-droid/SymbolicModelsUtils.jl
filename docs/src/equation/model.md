# Equation Model

This provides an environment for solving various equations models.
```@contents
Pages = ["model.md"]
Depth = 2:2
``` 

We use `Symbolics.jl` for handling symbolic equations. 

To simply generate an equation, we can follow from `Symbolics.jl`
```@example eqmodel
using SymbolicModelsUtils # hide
using Symbolics 
@syms x 
x^2 + 3x ~ 4x
```

But of course, if you are training an RL model, you will want to have equations generated automatically and randomly. For that you can check out [Dataset for Equation Model](@ref)

## Linear Model
Here, we support just two types of linear equations: `ax + b = c` and `a₁x + b₁ = a₂x + b₂` where the coefficients can either just be constants (simply integers or fractions), or symbolics, or both. 

```@repl eqmodel
eq = linear_eq(1, seed=123)
```

We can solve any linear equation model, by just transferring terms from one side of the equation to the other side. 
Here,` (1,1)` represents `(LHS, first term)` and then the function automatically handles whether it had to be subtracted or divided.
```@repl eqmodel
eq = linear_transport(eq, 1, 1)

eq = linear_transport(eq, 1, 1)
```

And to know if the equation has been solved call `linear_termination_status`, it checks whether there is `x` on the one side of the equation and `constant` on the other side.
```@repl eqmodel
linear_termination_status(eq)
```

For more examples see [Linear Model Examples](@ref)

## Quadratic Model
Quadratic equations of the form a₁x^2 + b₁x + c₁ ~ a₂x^2 + b₂x + c₂  where, constants can be number, fraction, symbolic

## Power Model

## Trignometric Model

## Functional Model

## API Reference
```@docs
linear_transport
linear_termination_status
```