# Dataset for Equation Model

This section provides an overview of how to generate dataset for Equation model. 

```@contents
Pages = ["dataset.md"]
Depth = 2:2
```

## Linear Dataset
There are two types of linear equations: `ax + b = c` and `a₁x + b₁ = a₂x + b₂`

```@repl dataset
using SymbolicModelsUtils # hide
linear_eq(5,1)
```

To change the coefficients just change the `type`.
```@repl dataset
eqs1 = linear_eq(1)  # ax + b = c (all numeric)
eqs2 = linear_eq(2)  # a₁x + b₁ = a₂x + b₂ (all numeric)
eqs3 = linear_eq(3)  # ax + b = c (symbolic constants)
eqs4 = linear_eq(4)  # a₁x + b₁ = a₂x + b₂ (symbolic constants)
eqs5 = linear_eq(5)  # Mixed types, random form
```

To know more about `type` of coefficients prefer [make_coeff](@ref).
Apart from the above types we can have fraction coefficients, using [fractional_linear_eq](@ref fractional_linear_eq). 

## Quadratic Dataset
```@docs
quadratic_eq
```

## Power Dataset
```@docs
power_eq
```

## Functional Dataset
By functional equations, we mean that it is combinations of some functions such that it is symbolically solvable. Now, if we take any random functions then it will be solvable only if we take composition of them without violating their respective domains. For that we can use

```@docs
functional_eq
```

But instead of choosing random functions, if we restrict ourself to one domain, like trignometric or exponential, then we can apply some simplification rule to reach a point where we can just take inverse and the problem will be solved. 