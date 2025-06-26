```@meta
CurrentModule = SymbolicModelsUtils
```

# SymbolicModelsUtils.jl
SymbolicModelsUtils.jl consists of various symbolics environments where a user or a model can perform \
symbolic operations as maths agree. The sole purpose of this module is to provide an environment in which \
users can try making an RL model learn. 

## Installation

To install SymbolicModelsUtils.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("https://github.com/JuliaSymbolics/SymbolicUtils.jl.git")
```

## Different Models

- Equation Model: for solving linear, quadratic, functional equations
- Integral Model: for solving integration problems

## Things to know
In all the models we by default consider that the missing variable is `x`. 
  
## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```