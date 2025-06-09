module SymbolicModelsUtils
using Symbolics
using SymbolicUtils
using SymbolicUtils: Rewriters, Prewalk, PassThrough

include("integral.jl")
export traverse_expr, show_full, ∫, ∂

include("Symbolics_func.jl")

include("Our_rules.jl")
export @birule, apply_∂, take_common

include("Equation model/dataset1.jl")
export make_coeff, linear_eq, fractional_linear_eq, quadratic_eq, power_eq, functional_eq, generate_rand_poly, polynomial_division, make_frac, partial_fraction, make_rand_func, extract_expression, change_variable

# For Linear Model
include("Equation model/linear_model.jl")
export linear_transport


# Check whether this module is working as expected or not
# Add the required modules (Random)
# Check whether I am importing different files from each other
# Check whether I am registering ∫, and ∂ more than once 

end
