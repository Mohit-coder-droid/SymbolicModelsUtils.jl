using Random

"""
Types of coefficient:
1. Number (can't be 0)
2. Symbolic with number coefficient
3. Symbolic with fraction coefficient
4. Fraction 
5. zero 

Examples:
```@repl 
    make_coeff([1,2,3,4,5],seed=123)
```
"""
function make_coeff(coeff_type::Vector{Int}, var_list::Vector{Num}=collect(@variables y z); rand_nu_range::Vector{Int64}=vcat(-128:-1, 1:127), seed=nothing, can_zero::Union{Nothing,Vector{Bool},BitVector}=nothing)
    tot = 0
    for i in eachindex(coeff_type)
        tot += (coeff_type[i] in [1, 2]) ? 1 : 2
    end
    rng = Xoshiro(seed)
    rand_without_0 = rand(rng, rand_nu_range, 1, tot)
    coeff = Vector{Any}(undef, length(coeff_type))

    if isnothing(can_zero)
        can_zero = falses(length(coeff_type))
    end

    rand_zero = rand(rng, sum(can_zero))
    tot_zero = 1
    tot = 1
    for i in 1:length(coeff)
        if can_zero[i]
            if rand_zero[tot_zero] < 0.2 # 20% probability to make the term 0
                coeff_type[i] = 5
            end
            tot_zero += 1
        end

        if coeff_type[i] == 1
            coeff[i] = rand_without_0[tot]
            tot += 1
        elseif coeff_type[i] == 2
            coeff[i] = rand_without_0[tot] * rand(rng, var_list)
            tot += 1
        elseif coeff_type[i] == 3
            coeff[i] = (rand_without_0[tot] // rand_without_0[tot+1]) * rand(rng, var_list)
            tot += 2
        elseif coeff_type[i] == 4
            coeff[i] = (rand_without_0[tot] // rand_without_0[tot+1])
            tot += 2
        elseif coeff_type[i] == 5
            coeff[i] = 0
        else
            error("Coefficient Type can be from 1 to 5 only")
        end
    end

    return coeff
end

"""
Swap LHS and RHS of an equation
"""
function swap_lhs_rhs(eq::Equation)
    return eq.rhs ~ eq.lhs
end

"""
Linear equation generator generates single equation

# Types:
1. ax + b = c (numeric constants)
2. a₁x + b₁ = a₂x + b₂ (numeric constants)  
3. ax + b = c (symbolic constants)
4. a₁x + b₁ = a₂x + b₂ (symbolic constants)
5. Mixed numeric/symbolic constants, random equation form (ax+b=c or a₁x+b₁=a₂x+b₂)
6. Custom coefficient types - specify [a₁, b₁, a₂, b₂] for a₁x+b₁=a₂x+b₂
"""
function linear_eq(type::Int, coeff_types::Vector{Int}=Int[1, 1, 1, 1], var_list::Vector{Num}=@variables y z; mix::Bool=false, seed=nothing)
    if mix
        type = 6
    end

    # Helper function to create simple form equation: ax + b = c
    function create_simple_eq(types::Vector{Int}; seed=nothing)  # types = [a,b,c]
        a, b, c = make_coeff(types, var_list, seed=seed)
        if (rand(Xoshiro(seed), Bool))
            return a * x + b ~ c
        else
            c ~ a * x + b
        end
    end

    # Helper function to create two-sided equation: a₁x + b₁ = a₂x + b₂
    function create_twosided_eq(types::Vector{Int}; seed=nothing)  # types = [a,b,c,d]
        a1, b1, a2, b2 = make_coeff(types, var_list, seed=seed)
        return a1 * x + b1 ~ a2 * x + b2
    end

    if type == 1
        return create_simple_eq([1, 1, 1], seed=seed)  # all numeric

    elseif type == 2
        return create_twosided_eq([1, 1, 1, 1], seed=seed)  # all numeric




    # make the coefficient of the leading term also symbolic, and ensure that it doesn't get 0 





    elseif type == 3
        if rand(Xoshiro(seed), Bool)
            return create_simple_eq([1, 2, 2], seed=seed)  # a=numeric, b,c=symbolic
        else
            return create_simple_eq([2, 2, 2], seed=seed)  # a=symbolic, b,c=symbolic
        end

    elseif type == 4
        if rand(Xoshiro(seed), Bool)
            return create_twosided_eq([1, 2, 1, 2], seed=seed)  # a₁,a₂=numeric, b₁,b₂=symbolic
        else
            return create_twosided_eq([1, 2, 2, 2], seed=seed)
        end

    elseif type == 5
        if rand(Xoshiro(seed), Bool)
            # Simple form with random coefficient types
            a_type = rand(Xoshiro(seed), [1, 2])  # coefficient of x must be non-zero
            b_type = rand(Xoshiro(seed), [1, 2, 4])
            c_type = rand(Xoshiro(seed), [1, 2, 4])
            return create_simple_eq([a_type, b_type, c_type], seed=seed)
        else
            # Two-sided form with random coefficient types
            a1_type = rand(Xoshiro(seed), [1, 2])
            a2_type = rand(Xoshiro(seed), [1, 2])
            b1_type = rand(Xoshiro(seed), [1, 2, 4])
            b2_type = rand(Xoshiro(seed), [1, 2, 4])
            return create_twosided_eq([a1_type, b1_type, a2_type, b2_type], seed=seed)
        end

    elseif type == 6
        if mix
            rand!(Xoshiro(seed), coeff_types, 1:5)
            coeff_types[1] = rand(Xoshiro(seed), 1:4)
        end

        # Custom coefficient types for a₁x + b₁ = a₂x + b₂
        @assert length(coeff_types) == 4 "For type 6, coeff_types must have 4 elements [a₁, b₁, a₂, b₂]"
        return create_twosided_eq([coeff_types[1], coeff_types[2], coeff_types[3], coeff_types[4]], seed=seed)

    else
        error("Invalid type. Must be 1, 2, 3, 4, 5, 6, or 6.")
    end
end

# it may happen that the function is calling itself again and again
# eqs1 = linear_eq(1, seed=12)  # ax + b = c (all numeric)
# eqs2 = linear_eq(2,seed=12)  # a₁x + b₁ = a₂x + b₂ (all numeric)
# eqs3 = linear_eq(3,seed=12)  # ax + b = c (symbolic constants)
# eqs4 = linear_eq(4,seed=12)  # a₁x + b₁ = a₂x + b₂ (symbolic constants)
# eqs5 = linear_eq(5,seed=12)  # Mixed types, random form
# eqs6 = linear_eq(6, [1, 2, 1, 4]; mix=true,seed=121)  # ax + b = c with custom types

"""Generates multiple linear equations"""
function linear_eq(nu::Int, type::Int, coeff_types::Vector{Int}=Int[1, 1, 1, 1], var_list::Vector{Num}=@variables y z; mix::Bool=false, seed=nothing)
    if mix
        type = 6
    end

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 1, nu)
    end

    eqs = Vector{Symbolics.Equation}(undef, nu)

    for i in 1:nu
        eqs[i] = linear_eq(type, coeff_types, var_list, mix=mix, seed=seed[i])
    end

    return eqs
end

# Usage examples:
# eqs1 = linear_eq(10, 1, seed=12)  # ax + b = c (all numeric)
# eqs2 = linear_eq(50, 2,seed=12)  # a₁x + b₁ = a₂x + b₂ (all numeric)
# eqs3 = linear_eq(50, 3,seed=12)  # ax + b = c (symbolic constants)
# eqs4 = linear_eq(50, 4,seed=12)  # a₁x + b₁ = a₂x + b₂ (symbolic constants)
# eqs5 = linear_eq(50, 5,seed=12)  # Mixed types, random form
# eqs6 = linear_eq(50, 6, [1, 2, 1, 4]; mix=true,seed=12)  # ax + b = c with custom types
"""
Fractional Linear equation generator

# Types
1. ax + b = c (numeric fraction)
2. a₁x + b₁ = a₂x + b₂ (numeric fraction)
3. ax + b = c (symbolic fraction)
4. a₁x + b₁ = a₂x + b₂ (symbolic fraction)
5. Mixed numeric/symbolic fraction, random equation form (ax+b=c or a₁x+b₁=a₂x+b₂)
"""
function fractional_linear_eq(type::Int, var_list::Vector{Num}=@variables y z; seed=nothing)
    # Helper function to create simple form equation: ax + b = c
    function create_simple_eq(types::Vector{Int}, mixed::Bool=false; seed=nothing)
        if mixed
            a, b, c = make_coeff([types[1], rand(Xoshiro(seed), [3, 4], 2)...], var_list,
                can_zero=[false, true, true], seed=seed)  # leading term can't be zero
        else
            a, b, c = make_coeff(types, var_list, can_zero=[false, true, true], seed=seed)
        end
        return a * x + b ~ c
    end

    # Helper function to create two-sided equation: a₁x + b₁ = a₂x + b₂
    function create_twosided_eq(types::Vector{Int}, mixed::Bool=false; seed=nothing)
        if mixed
            a1, a2, b1, b2 = make_coeff([types[1:2]..., rand(Xoshiro(seed), [3, 4], 2)...], var_list,
                can_zero=[false, false, true, true], seed=seed)
        else
            a1, a2, b1, b2 = make_coeff(types, var_list, can_zero=[false, false, true, true], seed=seed)
        end
        return a1 * x + b1 ~ a2 * x + b2
    end

    eqs = if type == 1
        create_simple_eq([3, 3, 3], seed=seed)
    elseif type == 2
        create_twosided_eq([3, 3, 3, 3], seed=seed)
    elseif type == 3
        create_simple_eq([4, 4, 4], seed=seed)
    elseif type == 4
        create_twosided_eq([4, 4, 4, 4], seed=seed)
    elseif type == 5
        # Randomly choose equation form and use mixed constants
        if rand(Xoshiro(seed), Bool)
            create_simple_eq([rand(Xoshiro(seed), [3, 4]), 4, 3], true, seed=seed)  # mixed=true
        else
            create_twosided_eq([rand(Xoshiro(seed), [3, 4], 2)..., 3, 3], true, seed=seed)  # mixed=true
        end
    else
        error("Invalid type. Must be 1, 2, 3, 4, or 5.")
    end

    return eqs
end

function fractional_linear_eq(nu::Int, type::Int, var_list::Vector{Num}=@variables y z; seed=nothing)
    eqs = Vector{Symbolics.Equation}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 1, nu)
    end

    for i in 1:nu
        eqs[i] = fractional_linear_eq(type, var_list, seed=seed[i])
    end

    return eqs

end

# eqs1 = fractional_linear_eq(50,1,seed=123)
# eqs2 = fractional_linear_eq(50,2,seed=123)
# eqs3 = fractional_linear_eq(50,3,seed=123)
# eqs4 = fractional_linear_eq(50,4,seed=123)
# eqs4 = fractional_linear_eq(50,5,seed=123)


# Why not we can apply direct quadratic formula to solve this quadratic equation (In symbolic calculations, we don't care about the sign inside the square root, until we aren't sure about the range of our variables) 

"""
Generates random quadratic equations of the form `a₁x^2 + b₁x + c₁ ~ a₂x^2 + b₂x + c₂`  where, constants can be number, fraction, symbolic

    quadratic_eq(nu::Int, type::Vector{Int}=[1, 1, 1, 1, 1, 1], var_list=@variables y z; mix::Bool=false, seed=nothing)

# Arguments
- `nu::Int`: number of equations
- `type::Vector{Int}`: determines whether coefficients in the equations are number, fraction, or symbolic on the basis of `make_coeff()`
- `var_list`: variables to be used as a symbolic constants 
- `mix::Bool`: makes `type` random 
- `seed` 

# Examples
```@example
quadratic_eq(type=[1,2,3,4,4,5], seed=123)
```
"""
function quadratic_eq(type::Vector{Int}=[1, 1, 1, 1, 1, 1], var_list=@variables y z; mix::Bool=false, seed=nothing)
    @assert length(type) == 6 "Length of type must be 6"

    if mix
        rng = Xoshiro(seed)
        rand!(rng, type, 1:5)  # modifies the actual type
        type[1] = rand(rng, 1:4)  # leading term can't be 0
    end

    a₁, b₁, c₁, a₂, b₂, c₂ = make_coeff(type, var_list, seed=seed, can_zero=Bool.([0, 1, 1, 1, 1, 1]))

    return a₁ * x^2 + b₁ * x + c₁ ~ a₂ * x^2 + b₂ * x + c₂
end

# make_coeff([1, 1, 1, 1, 1, 1], seed=6, can_zero=Bool.([0, 1, 1, 1, 1, 1]))

# nu = 50
# eqs = Vector{Symbolics.Equation}(undef, nu)
# for i in 1:nu
#     eqs[i] = quadratic_eq([1, 1, 1, 1, 1, 1], seed=nothing, mix=false)
# end
# eqs

function quadratic_eq(nu::Int, type::Vector{Int}=[1, 1, 1, 1, 1, 1], var_list=@variables y z; mix::Bool=false, seed=nothing)
    @assert length(type) == 6 "Length of type must be 6"

    eqs = Vector{Symbolics.Equation}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), nu)
    end

    for i in 1:nu
        eqs[i] = quadratic_eq(copy(type), var_list, mix=mix, seed=seed[i])
    end

    return eqs
end
# quadratic_eq(10, [1, 2, 3, 5, 1, 4]; seed=123)
# qe = quadratic_eq(10, seed=123)

# Check for the domain error possiblity
"""
Generates equations involving powers

# Types
1. a*x^(b) + c = d
2. a₁*x^(b₁) = a₂*x^(b₂)
"""
function power_eq(type::Int, coeff_type::Vector{Int}=[1, 1, 1, 1]; mix::Bool=false, seed=nothing)
    @assert length(coeff_type) == 4 "Length of coeff_type must be equal to 4"

    if mix
        rand!(Xoshiro(seed), coeff_type, 1:5)
        coeff_type[1] = rand(Xoshiro(seed), 1:4)
        type = rand(Xoshiro(seed), 1:2)
    end

    if type == 1
        coeff = make_coeff(coeff_type, seed=seed, can_zero=[false, false, true, false])
        return coeff[1] * x^(coeff[2]) + (coeff[3]) ~ (coeff[4])
    elseif type == 2
        coeff = make_coeff(coeff_type, seed=seed, can_zero=[false, false, false, false])
        return coeff[1] * x^(coeff[2]) ~ coeff[3] * x^(coeff[4])
    else
        error("Type can be only 1 or 2")
    end
end

function power_eq(nu::Int, type::Int, coeff_type::Vector{Int}=[1, 1, 1, 1]; mix::Bool=false, seed=nothing)
    @assert length(coeff_type) == 4 "Length of coeff_type must be equal to 4"
    eqs = Vector{Symbolics.Equation}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 1, nu)
    end

    for i in 1:nu
        eqs[i] = power_eq(type, coeff_type, seed=seed[i], mix=mix)
    end

    return eqs
end

# power_eq(10, 1, seed=123)
# power_eq(10, 2, seed=123)
# power_eq(10, 2; mix=true, seed=123)

# Include [sin, cos, sec, cosec, tan, cot, log, exp, sqrt, power_n, power_1_n, rational_power, sinh, cosh,sech, cosech, tanh, coth,atan,asin,acos,asinh,acosh,atanh,acoth,asech,acsch, ]
# Does putting function into linear, quad, equ does make sense, as those things can easily be solved using susbstitution
# I think substituion model has to be made differently
func_list = [sin, cos, sec, csc, tan, cot, log, exp, sqrt, sinh, cosh, sech, csch, tanh, coth, atan, asin, acos, asinh, acosh, atanh, acoth, asech, acsch]

# In this equations check for domain errors
"""
1. Direct inverse: f(x) = const 
2. Composition of two func: (g ∘ f)(x) = const 
3. Composition of three func: (h ∘ g ∘ f)(x) = const
"""
function functional_eq(depth::Int, coeff_type::Int=1; mix::Bool=false, seed=nothing)
    if mix
        coeff_type = rand(Xoshiro(seed), 1:5)
        depth = rand(Xoshiro(seed), 1:3)
    end

    return reduce(∘, rand(Xoshiro(seed), func_list, depth))(x) ~ make_coeff([coeff_type], seed=seed)[1]
end

function functional_eq(nu::Int, depth::Int, coeff_type::Int; mix::Bool=false, seed=nothing)
    eqs = Vector{Symbolics.Equation}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 1, nu)
    end

    for i in 1:nu
        eqs[i] = functional_eq(depth, coeff_type; mix=mix, seed=seed[i])
    end

    return eqs
end
# functional_eq(2,2,1, seed=12)

function generate_rand_poly(max_deg::Int, coeff_type::Vector{Int}; mix::Bool=false, min_deg::Int=0, seed=nothing)
    @assert length(coeff_type) == (max_deg + 1) "Length of coeff_type should be (max_deg+1)"

    if mix
        rand!(Xoshiro(seed), coeff_type, 1:5)
        coeff_type[min_deg+1] = rand(Xoshiro(seed), 1:4)
    end

    can_zero = trues(length(coeff_type))
    can_zero[end] = false
    terms = make_coeff(coeff_type, can_zero=can_zero, seed=seed)

    for deg in 0:max_deg
        terms[deg+1] = terms[deg+1] * x^deg
    end

    return reduce(+, terms)
end

# generate_rand_poly(3, [1, 1, 1, 1], mix=true, seed=123)

# at max power of remainder and quotient can be 3, and min can be 1
"""
Give polynomial division multiplication

This dataset has been wrongly created, as P(x) = h(x) * q(x) + r(x)
"""
function polynomial_division(nu::Int, coeff_type::Vector{Vector{Int}}; mix::Bool=false, min_deg::Int=1, max_deg::Int=2, seed=nothing)
    eqs = Vector{Vector{Num}}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, 3 * nu + 2)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 1, 3 * nu + 2)
    end

    for i in 1:nu
        r = generate_rand_poly(max_deg, coeff_type[3]; mix=mix, min_deg=min_deg, seed=seed[3*i])
        q = generate_rand_poly(max_deg, coeff_type[2]; mix=mix, min_deg=min_deg, seed=seed[3*i+1])
        h = generate_rand_poly(max_deg, coeff_type[1]; mix=mix, min_deg=min_deg, seed=seed[3*i+2])

        eqs[i] = [h, q, r, simplify(h * q + r, expand=true)]
    end
    return eqs
end

# polynomial_division(5, [[1, 1, 1], [1, 1, 1],[1, 1, 1]], seed=123)

# make_coeff(2, rand_nu_range=vcat(-9:-1, 1:9))

"""
Linear fraction: c // (ax + b)
Quadratic fraction: (dx + e) // (ax^2 + bx + c)
"""
function make_frac(coeff_type, is_quad::Bool=false; seed=nothing)
    if !is_quad
        @assert length(coeff_type) >= 3 "Length of coeff_type for linear fraction must be more than 3"
        a, b, c = make_coeff(coeff_type[1:3], can_zero=Bool.([0, 1, 0]), seed=seed)
        return c // (a * x + b)
    else
        @assert length(coeff_type) == 5 "Length of coeff_type for quadratic fraction must be 5"
        a, b, c, d, e = make_coeff(coeff_type, can_zero=Bool.([0, 1, 1, 1, 0]), seed=seed)
        return (d * x + e) // (a * x^2 + b * x + c)
    end
end

# make_frac([1, 1, 1], seed=123)
# make_frac([1, 5, 1, 1, 2], true, seed=123)

"""
Give partial fraction problems 

Types:
1. Two linear Fractions
2. One linear, One quadratic Fractions
3. three linear Fractions
"""
function partial_fraction(type::Int, coeff_type; mix::Bool=false, seed=nothing)
    if mix
        type = rand(Xoshiro(seed), 1:3)
        mat = rand(Xoshiro(seed), [1, 2], 3, 5)
        coeff_type = [collect(i) for i in eachrow(mat)]
    end

    if isnothing(seed)
        seed = Vector{Any}(nothing, 3)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), 3)
    end

    if type == 1
        frac1, frac2 = make_frac(coeff_type[1], seed=seed[1]), make_frac(coeff_type[2], seed=seed[2])
        eqs = [simplify_fractions(frac1 + frac2), frac1, frac2]
    elseif type == 2
        frac1, frac2 = make_frac(coeff_type[1], seed=seed[1]), make_frac(coeff_type[2], true, seed=seed[2])
        eqs = [simplify_fractions(frac1 + frac2), frac1, frac2]
    elseif type == 3
        frac1, frac2, frac3 = make_frac(coeff_type[1], seed=seed[1]), make_frac(coeff_type[2], seed=seed[2]), make_frac(coeff_type[3], seed=seed[3])
        eqs = [simplify_fractions(frac1 + frac2 + frac3), frac1, frac2, frac3]
    else
        error("Type can be 1,2,3")
    end

    return eqs
end

# partial_fraction(1, [[1, 1, 1], [1, 1, 1]], mix=true, seed=123)
# partial_fraction( 2, [[1, 1, 1], [1, 1, 1, 1, 1]], seed=123)
# partial_fraction(3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], seed=123)

function partial_fraction(nu::Int, type::Int, coeff_type; mix::Bool=false, seed=nothing)
    eqs = Vector{Vector{Num}}(undef, nu)

    if isnothing(seed)
        seed = Vector{Any}(nothing, nu)
    else
        seed = rand(Xoshiro(seed), (-128:-1; 1:128), nu)
    end

    for i in 1:nu
        eqs[i] = partial_fraction(type, coeff_type, mix=mix, seed=seed[i])
    end

    return eqs
end

# partial_fraction(5, 1, [[1, 1, 1], [1, 1, 1]], seed=123)
# partial_fraction(5, 2, [[1, 1, 1], [1, 1, 1, 1, 1]], seed=123)
# partial_fraction(5, 3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]],mix=true, seed=123)


# To extract some expression 
# like sin(2x) ∼ 2sin(x)cos(x) => sin(x)cos(x) ∼ sin(2x)//2   [extract sin(x)cos(x)]
# At one of the type just limit the composition of functions as 
# There can be at max 4 func on each side and at max 2 depth composition
# should number is also to be counted as func

# Let m(x) be the function that has to be extracted 
# Then generate any random expression, and just put m anywhere on that expression 

# Any variable that is being replaced can be replaced by composition of atleast two functions or at max three functions. For composition we can use basic_diadic
basic_diadic = [+, -, /, *, ^]

expr_types = [linear_eq, fractional_linear_eq, quadratic_eq, power_eq, functional_eq, partial_fraction]

# fractional_linear_eq(50,5, seed=123)
# quadratic_eq(50, mix=true, seed=123)
# power_eq(50, mix=true,seed=123)
# functional_eq(50, mix=true,seed=123)
# partial_fraction(50, 1, [[2, 2, 2], [2, 2, 2]], seed=123)
# linear_eq(10, mix=true, seed=123)

"""
Types:
1. operation(func1, func2)
2. operation1(func1, operation2(func2, func3))

operation are basic_diadic = [+, -, /, *, ^]
"""
function make_rand_func(type::Int=1; seed=nothing)
    if type == 1
        f1, f2 = rand(func_list, 2)
        op = rand(Xoshiro(seed), basic_diadic)

        return op(f1(x), f2(x))
    elseif type == 2
        op1, op2 = rand(Xoshiro(seed), basic_diadic, 2)
        f1, f2, f3 = rand(Xoshiro(seed), func_list, 3)

        return op1(f1(x), op2(f2(x), f3(x)))
    else
        error("Type can be 1 or 2")
    end
end

# make_rand_func(2, seed=123)

# I think this function is not completed
"""
Types:
1. Just composition of multiplication
2. Composition of multiplication, addition, division
3. Composition using power (can be rational)
4. Composition of functions together
"""
function extract_expression(nu::Int=10, type::Int=1; put_m::Bool=false, seed=nothing)
    # Make mix default in all the expr_types
    linear_eq_(nu) = linear_eq(nu, mix=true, seed=seed)
    fractional_linear_eq_(nu) = fractional_linear_eq(nu, 5, seed=seed)
    quadratic_eq_(nu) = quadratic_eq(nu, mix=true, seed=seed)
    power_eq_(nu) = power_eq(nu, mix=true, seed=seed)
    functional_eq_(nu) = functional_eq(nu, mix=true, seed=seed)
    partial_fraction_(nu) = partial_fraction(nu, 1, [[2, 2, 2], [2, 2, 2]], mix=true, seed=seed)

    expr_types = [linear_eq_, fractional_linear_eq_, quadratic_eq_, power_eq_, functional_eq_, partial_fraction_]

    function insert_randomly!(expr, m, rng)
        # Navigate the tree randomly, and at each location give it a chance to either put m there and return, or to go further
        # if hit a dead end, then just put it there
        if !iscall(expr)
            op = rand(rng, basic_diadic)
            return op(m, expr)
        end

        new_args = arguments(expr)  # this code make reference of it, updating this will automatically update the expr

        loc = rand(1:length(new_args))
        if rand(rng, Bool)  # put m here
            op = rand(rng, basic_diadic)
            new_args[loc] = op(m, new_args[loc])
        else
            new_args[loc] = insert_randomly!(new_args[loc], m, rng)
        end
        return Symbolics.operation(expr)(new_args...)
    end

    if type == 6
        eqs = Vector{Equation}(undef, nu)
        eqs_ = expr_types[type](nu)
        for i in eachindex(eqs)
            eqs[i] = eqs_[i][1] ~ eqs_[i][2] + eqs_[i][3]
        end
    else
        eqs = expr_types[type](nu)
    end

    if put_m
        m_values = Vector{Num}(undef, nu)
    end

    for i in eachindex(eqs)
        rng = Xoshiro(seed)

        variables = filter(!isequal(Symbolics.value(x)), Symbolics.get_variables(eqs[i]))
        subsi = Dict{SymbolicUtils.BasicSymbolic,Num}()
        for v in variables
            subsi[v] = make_rand_func(rand(rng, 1:2), seed=seed)
        end

        eqs[i] = Symbolics.substitute(eqs[i], subsi)

        if rand(Bool)  # add m on lhs 
            insert_randomly!(eqs[i].lhs, m, rng)
        else
            insert_randomly!(eqs[i].rhs, m, rng)
        end

        if put_m  # put any random value in place of m
            m_values[i] = make_rand_func(rand(rng, 1:2), seed=seed)
            eqs[i] = substitute(eqs[i], Dict(m => m_values[i]))
        end
    end

    if put_m
        return eqs, m_values
    end

    return eqs
end

# extract_expression(20, 6)

# Valid just for trignometric substituions
@variables x
subsi_func = [sin(x), cos(x), tan(x), cot(x), sec(x), csc(x), sin(2x), cos(2x), tan(2x)]
expr_func = [sin(x), cos(x), tan(x), cot(x), sec(x), csc(x), sin(2x), cos(2x), tan(2x), sin(3x), cos(3x), tan(3x)]
# make_coeff(2, rand_nu_range=vcat(-10:-1, 1:10))

# rand(subsi_func, 3)

# This function is also incomplete
function change_variable(nu::Int=1; expr_func::Vector{Num})
    eqs = Vector{Vector{Num}}(undef, nu)
    function make_rand_expr(max_expr_func::Int=5)
        funcs = rand(expr_func, rand(2:max_expr_func))
        op = rand(basic_diadic, length(funcs) - 1)

        expr = funcs[end]
        for i in (length(funcs)-1):-1:1
            if rand() < 0.3  # discard this function and put constant there
                expr = op[i](make_coeff([rand(1:4)], rand_nu_range=vcat(-10:-1, 1:10)), expr)[1]
            else
                expr = op[i](funcs[i], expr)
            end
        end

        return expr
    end

    for i in 1:nu
        eqs[i] = [rand(subsi_func), make_rand_expr()]
    end

    return eqs
end

# change_variable(10; expr_func=expr_func)

# One more type of dataset is left
# Here, it's not just direct already learned transformations, but also application of some already pre-defined rules will be required to handle this task
# This is much more same as extract_expression() function 
# Come back to this once I am more clarified about what this dataset is about 
function extract_expression_using_rules()

end

# One more type of dataset that can be created is that given some rules, try to devise some required expression out of it. 
# Many times it happens that we have some theories and we have some experimental data, and by fitting data we can figure out some formula but we don't able to bring that formula from theory. So, in that case this model can be helpful 
function expression_from_rule()

end


# How the action will be taking place
# There are two parts LHS and RHS, make an action to transfer expression from lhs to rhs, only when operation +, -, *, / are present
# otherwise apply inverse of the existing function both sides, it's just linear equation so no inverse of any function 
# Also make an action to multiply/add/sub/divide some expression both sides of an equation
