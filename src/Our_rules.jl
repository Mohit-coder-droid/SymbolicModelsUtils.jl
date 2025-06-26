########################## Predicates #########################################

is_pos_int(x) = isinteger(x) && x > 0
is_neg_int(x) = isinteger(x) && x < 0
is_int_gt_one(x) = isinteger(x) && x > 1
is_pos(x) = x > 0
is_neg(x) = x < 0
is_neg_one(x) = (x ≈ -1)
is_pos_half(x) = (x ≈ 0.5)
is_neg_half(x) = (x ≈ -0.5)
is_abs_half(x) = (x ≈ 0.5 || x ≈ -0.5)

have_addition(x) = (iscall(x) && typeof(operation(x))==typeof(+))
have_multiplication(x) = (iscall(x) && typeof(operation(x))==typeof(*))
have_division(x) = (iscall(x) && typeof(operation(x))==typeof(/))

########################## Bi Directional Macro #########################################

"""
Struct to hold information about equation and rule in @birule
"""
struct BiRule
    eq::Expr
    rule
end

function (br::BiRule)(x)
    br.rule(x)
end

function Base.show(io::IO, bir::BiRule)
    Base.print(io, bir.eq)
end

"""
Works in both direction 

And runs prewalk by default
"""
macro birule(expr)
    @assert expr.head == :call && expr.args[1] == :(~)
    lhs, rhs = expr.args[2], expr.args[3]

    return esc(quote
        let
            forward = SymbolicUtils.@acrule $lhs => $rhs
            reverse = SymbolicUtils.@acrule $rhs => $lhs

            # Should not apply prewalk as it will lead to confusion for the model to know exactly on what the rule is getting applied, and what effect it is having on that 
            # forward_walker = Prewalk(PassThrough($forward))
            # reverse_walker = Prewalk(PassThrough($reverse))

            rule = x -> begin
                result = forward(x)  # walker doesn't return nothing, they return the same expression
                result === nothing ? reverse(x) : result
            end

            BiRule($(QuoteNode(expr)), rule)
        end
    end)
end

########################## Transformation Rules ###############################

trigs_rules = [@birule tan(~x) ~ sin(~x) / cos(~x) # 1
    @birule sec(~x) ~ one(~x) / cos(~x) # 2
    @birule csc(~x) ~ one(~x) / sin(~x) # 3
    @birule cot(~x) ~ cos(~x) / sin(~x) # 4
    @birule sin(~x)^2 ~ 1 - cos(~x)^2# 5
    @birule sec(~x)^2 ~ 1 + tan(~x)^2  #6 
    @birule csc(~x)^2 ~ 1 + cot(~x)^2 #7 
    @birule sin(2*~x) ~ 2*sin(~x)*cos(~x)  # 8
    @birule cos(2*~x) ~ 2*(^(cos(~x), 2)) - 1
    @rule sin(~n::is_int_gt_one * ~x) => sin((~n - 1) * ~x) * cos(~x) +
                                        cos((~n - 1) * ~x) * sin(~x)
    @rule cos(~n::is_int_gt_one * ~x) => cos((~n - 1) * ~x) * cos(~x) -
                                        sin((~n - 1) * ~x) * sin(~x)
    @rule tan(~n::is_int_gt_one * ~x) => (tan((~n - 1) * ~x) + tan(~x)) /
                                        (1 - tan((~n - 1) * ~x) * tan(~x))
    @rule csc(~n::is_int_gt_one * ~x) => sec((~n - 1) * ~x) * sec(~x) *
                                        csc((~n - 1) * ~x) * csc(~x) /
                                        (sec((~n - 1) * ~x) * csc(~x) +
                                            csc((~n - 1) * ~x) * sec(~x))
    @rule sec(~n::is_int_gt_one * ~x) => sec((~n - 1) * ~x) * sec(~x) *
                                        csc((~n - 1) * ~x) * csc(~x) /
                                        (csc((~n - 1) * ~x) * csc(~x) -
                                            sec((~n - 1) * ~x) * sec(~x))
    @rule cot(~n::is_int_gt_one * ~x) => (cot((~n - 1) * ~x) * cot(~x) - 1) /
                                        (cot((~n - 1) * ~x) + cot(~x))
    @birule ~n / sin(~x) ~ ~n * csc(~x)
    @birule ~n / cos(~x) ~ ~n * sec(~x)
    @birule ~n / tan(~x) ~ ~n * cot(~x)
    @birule ~n / csc(~x) ~ ~n * sin(~x)
    @birule ~n / sec(~x) ~ ~n * cos(~x)
    @birule ~n / cot(~x) ~ ~n * tan(~x)

    @birule ~n / ^(sin(~x), ~k) ~ ~n * ^(csc(~x), ~k)
    @birule ~n / ^(cos(~x), ~k) ~ ~n * ^(sec(~x), ~k)
    @birule ~n / ^(tan(~x), ~k) ~ ~n * ^(cot(~x), ~k)
    @birule ~n / ^(csc(~x), ~k) ~ ~n * ^(sin(~x), ~k)
    @birule ~n / ^(sec(~x), ~k) ~ ~n * ^(cos(~x), ~k)
    @birule ~n / ^(cot(~x), ~k) ~ ~n * ^(tan(~x), ~k)
    @birule sin(~x + ~y) ~ sin(~x) * cos(~y) + cos(~x) * sin(~y)
    @birule cos(~x + ~y) ~ cos(~x) * cos(~y) - sin(~x) * sin(~y)
    @birule tan(~x + ~y) ~ (tan(~x) + tan(~y)) / (1 - tan(~x) * tan(~y))
    @birule csc(~x + ~y) ~ sec(~x) * sec(~y) * csc(~x) * csc(~y) /
                            (sec(~x) * csc(~y) + csc(~x) * sec(~y))
    @birule sec(~x + ~y) ~ sec(~x) * sec(~y) * csc(~x) * csc(~y) /
                            (csc(~x) * csc(~y) - sec(~x) * sec(~y))
    @birule cot(~x + ~y) ~ (cot(~x) * cot(~y) - 1) / (cot(~x) + cot(~y))
    @birule sin(~x - ~y) ~ sin(~x) * cos(~y) - cos(~x) * sin(~y)
    @birule cos(~x - ~y) ~ cos(~x) * cos(~y) + sin(~x) * sin(~y)
    @birule tan(~x - ~y) ~ (tan(~x) - tan(~y)) / (1 + tan(~x) * tan(~y))
    @birule csc(~x - ~y) ~ sec(~x) * sec(~y) * csc(~x) * csc(~y) /
                            (sec(~x) * csc(~y) - csc(~x) * sec(~y))
    @birule sec(~x - ~y) ~ sec(~x) * sec(~y) * csc(~x) * csc(~y) /
                            (csc(~x) * csc(~y) + sec(~x) * sec(~y))
    @birule cot(~x - ~y) ~ (cot(~x) * cot(~y) + 1) / (cot(~x) - cot(~y))

    @rule sin(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? 2*sin(~n/2 * ~x)*cos(~n/2 * ~x) : nothing
    @rule cos(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? 2*cos(~n/2 * ~x)^2 - 1 : nothing
    @rule tan(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? 2*tan(~n/2 * ~x)/(1 - tan(~n/2 * ~x)^2) : nothing
    @rule cot(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? (cot(~n/2 * ~x)^2 - 1)/(2*cot(~n/2 * ~x)) : nothing
    @rule sec(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? sec(~n/2 * ~x)^2/(2 - sec(~n/2 * ~x)^2) : nothing
    @rule csc(~n * ~x) => ((~n % 2)==0 && ~n > 0) ? sec(~n/2 * ~x)*csc(~n/2 * ~x)/2 : nothing

    @rule sin(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? 3*sin(~n/3 * ~x) - 4*sin(~n/3 * ~x)^3 : nothing
    @rule cos(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? 4*cos(~n/3 * ~x)^3 - 3*cos(~n/3 * ~x) : nothing        
    @rule tan(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? (3*tan(~n/3 * ~x) - tan(~n/3 * ~x)^3)/(1 - 3*tan(~n/3 * ~x)^2) : nothing        
    @rule cot(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? (3*cot(~n/3 * ~x) - cot(~n/3 * ~x)^3)/(1 - 3*cot(~n/3 * ~x)^2) : nothing        
    @rule sec(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? sec(~n/3 * ~x)^3/(4 - 3*sec(~n/3 * ~x)^2) : nothing        
    @rule csc(~n * ~x) => ((~n % 3)==0 && ~n > 0) ? csc(~n/3 * ~x)^3/(3*csc(~n/3 * ~x)^2 - 4) : nothing

    @birule tanh(~x) ~ sinh(~x) / cosh(~x)
    @birule sech(~x) ~ one(~x) / cosh(~x)
    @birule csch(~x) ~ one(~x) / sinh(~x)
    @birule coth(~x) ~ cosh(~x) / sinh(~x)
    @birule exp(~x) * exp(~y) ~ exp(~x + ~y)]

#
pow_rules = [
    # I think I need to find another way of tackling this problem 
    @birule /(~n,^(~a, ~x)) ~ ~n * ^(~a, -1*~x)
    @birule /(^(~a, ~x), ^(~a, ~y)) ~ ^(~a, ~x-~y)
    @birule *(^(~a, ~x), ^(~a, ~y)) ~ ^(~a, ~x+~y)

    # power of exp
    @birule /(~n1, exp(~n2 * ~x)) ~ *(~n1, exp(-1 * ~n2 * ~x))
    @birule exp(~n1 * ~x) ^ ~n2 ~ exp(~n1 * ~n2 * ~x)
    @birule exp(~n * ~x) ~ ^(exp(~x), ~n)
    @birule exp(~x) * exp(~y) ~ exp(~x + ~y)
    @birule exp(~x) / exp(~y) ~ exp(~x - ~y)
]

# Check again
# pow_rules[1](2/2^x)
# pow_rules[2](2^(x-y))
# pow_rules[2](2^(x)/2^y)
# pow_rules[6](exp(2t))

log_rules = [
    @birule log(~a*~b) ~ log(~a) + log(~b)
    @birule log(~a/~b) ~ log(~a) - log(~b)
    @birule log(^(~a, ~x)) ~ ~x * log(~a)
]
# log_rules[2](log(x)-log(y))  # check why the inverse part is not working 

# yup = @rule log(~a) - log(~b) => log(~a / ~b)
# yup = @rule -log(~a) + log(~b) => log(~a / ~b)
# yup(log(x)-log(y))

   ########################## Integration Rules ###############################
int_rules = [
    @rule ∫(^(~x,~n), ∂(~x)) => (~n!=-1) ? ^(~x, ~n+1) / (~n+1) : nothing
    @rule ∫(exp(~x), ∂(~x)) => exp(~x)
    @rule ∫(/(1,~x), ∂(~x)) => log(~x)
    @rule ∫(^(~a, ~x), ∂(~x)) => ^(~a, ~x) / log(~a)

    # Trigonometric function
    @rule ∫(sin(~x), ∂(~x)) => -cos(~x)
    @rule ∫(cos(~x), ∂(~x)) => sin(~x)
    @rule ∫(sec(~x)^2, ∂(~x)) => tan(~x)
    @rule ∫(csc(~x)^2, ∂(~x)) => -cot(~x)
    @rule ∫(sec(~x)*tan(~x), ∂(~x)) => sec(~x)
    @rule ∫(cot(~x)*csc(~x), ∂(~x)) => -csc(~x)
    @rule ∫(sinh(~x), ∂(~x)) => cosh(~x)
    @rule ∫(cosh(~x), ∂(~x)) => sinh(~x)
    @rule ∫(tan(~x), ∂(~x)) => -log(cos(~x))
    @rule ∫(cot(~x), ∂(~x)) => log(sin(~x))
    @rule ∫(log(~x), ∂(~x)) => -~x + ~x * log(~x)

    # Inverse Trigonometric function
    @rule ∫(atan(~x), ∂(~x)) => ~x * atan(~x) - 1 / 2 * log(1 + (~x)^2)
    @rule ∫(asin(~x), ∂(~x)) => ~x * asin(~x) + sqrt(1 - (~x)^2)
    @rule ∫(acos(~x), ∂(~x)) => ~x * acos(~x) - sqrt(1 - (~x)^2)
    @rule ∫(asinh(~x), ∂(~x)) => ~x * asinh(~x) - sqrt(~x^2 + 1)
    @rule ∫(acosh(~x), ∂(~x)) => ~x * acosh(~x) - sqrt(~x^2 - 1)  # for x≥1
    @rule ∫(atanh(~x), ∂(~x)) => ~x * atanh(~x) + 1/2 * log(1 - ~x^2)  # for |x|<1
    @rule ∫(acoth(~x), ∂(~x)) => ~x * acoth(~x) + 1/2 * log(~x^2 - 1)  # for |x|>1
    @rule ∫(asech(~x), ∂(~x)) => ~x * asech(~x) + asin(~x)  # for 0<x≤1
    @rule ∫(acsch(~x), ∂(~x)) => ~x * acsch(~x) - asin(1/~x)  # for 0<x≤1

    # Some more standard integrals
    @rule ∫(1 / (sqrt((~x)^2 + (~a)^2)), ∂(~x)) => log(abs(~x + sqrt((~x)^2 + (~a)^2)))
    @rule ∫(1 / (sqrt((~x)^2 + (~a)^2)), ∂(~x)) => asinh(x/a)
    @rule ∫(1 / (sqrt((~x)^2 - (~a)^2)), ∂(~x)) => log(abs(~x + sqrt((~x)^2 - (~a)^2)))
    @rule ∫(1 / (sqrt((~x)^2 - (~a)^2)), ∂(~x)) => acosh(x/a)

    @rule ∫(1 / ((~a)^2 - (~x)^2),∂(~x)) => log(abs((~a+~x) / (~a-~x))) / (2*~a)
    @rule ∫(1 / ((~a)^2 - (~x)^2),∂(~x)) => atanh(~x/~a)
    @rule ∫(1 / ((~x)^2 - (~a)^2),∂(~x)) => log(abs((~x - ~a) / (~a+~x))) / (2*~a)
    @rule ∫(1 / ((~x)^2 - (~a)^2),∂(~x)) => acoth(~x/~a)
]


function expand_integral_sum(expr)
    check_applicable = @rule ∫(~uv, ∂(~x)) => (~uv, ∂(~x))

    appl = check_applicable(expr)
    if isnothing(appl) || !have_addition(appl[1]) return nothing end

    u = Vector{Any}()
    for arg in arguments(appl[1])
        push!(u, ∫(arg, appl[2]))
    end

    reduce(+, u)
end

# expr = ∫(sin(x)*tan(x) + cos(x) + 1 , ∂(x))
# expand_integral_sum(expr)

basic_int_rules = [
    @birule ∫(~uv, ∂(~x)) ~ ∫(~uv,∂(~x))
]
# basic_int_rules[1](∫(sin(x),∂(x)))

heur_int_rules = [
    @rule ∫(~b / ^(~x, ~n), ∂(~x)) => (~n!=1) ? ^(~x, -1*~n+1) / (-1*~n+1) : nothing
]

# int_rules[end](∫(acos(2x), ∂(2x)))
# int_rules[4](∫((2)^(-x), ∂(x)))
# int_rules[4](∫((2)^(x), ∂(x)))


# heur_int_rules[1](∫(1 / (t^2), ∂(t)))


"""Removes the constant from integration"""
function const_out(expr)
    check_applicable = @rule ∫(~uv, ∂(~x)) => (~uv, ∂(~x))

    appl = check_applicable(expr)
    if appl === nothing
        return nothing
    end

    if typeof(appl[1])!=typeof(SymbolicUtils.BasicSymbolic{Real})  # this means it must be number
        return appl[1]*arguments(appl[2])[1]
    end

    num_den = Vector{Vector{Any}}()
    for i in 1:4
        push!(num_den, [1])
    end  #[[num(x), !num(x), denom(x), !denom(x)]]
    func = [numerator, denominator]

    function push_num_den!(occuring_term, term, i::Int)
        if occursin(occuring_term, term)
            push!(num_den[2*(i-1)+1], term)
        else
            push!(num_den[2*(i-1)+2], term)
        end
    end

    for i in 1:2
        f = func[i]
        if iscall(f(appl[1]))
            if typeof(operation(f(appl[1]))) == typeof(*)
                for arg in arguments(f(appl[1]))
                    push_num_den!(arguments(appl[2])[1], arg, i)
                end
            else  # if terms aren't connected by * nor by /, then just consider it as a single term
                push_num_den!(arguments(appl[2])[1], f(appl[1]),i)
            end
        else
            push_num_den!(arguments(appl[2])[1], f(appl[1]),i)
        end
    end


    if (length(num_den[1]) * length(num_den[3]) == 1)
        # means ∫ is totally const, so it's like ∫const*∂(x) = const*x
        if iscall(arguments(appl[2])[1])  # ~x shouldn't be callable like sin(x)
            return nothing
        end

        return *(
            /(reduce(*, num_den[2]), reduce(*, num_den[4])),
            arguments(appl[2])[1]
        )
    end

    return *(
        /(reduce(*, num_den[2]), reduce(*, num_den[4])),
        ∫(
            /(reduce(*, num_den[1]), reduce(*, num_den[3])),
            appl[2]
        )
    )

end

# expr = Symbolics.value(∫(y * tan(x) / sin(x), ∂(x)))
# expr = Symbolics.value(∫(sin(y) * cos(y) * sin(x), ∂(x)))
# expr = Symbolics.value(∫(y, ∂(x)))
# expr = Symbolics.value(∫(1, ∂(x)))

# const_out(expr)


"""
To check whether the operator exists in the expression or not, if it does then it will return whole operator tree (first instance), if it doesn't then nothing
"""
function occursin_operator(operator,expr)
    if iscall(expr)
        args, op = arguments(expr), operation(expr)
        if (op == operator)
            return expr
        end

        for arg in args
            found = occursin_operator(operator,arg)
            if !isnothing(found)
                return found
            end
        end
    end

    return nothing
end

"""
∫() takes two arguments, consider a case ∫(exp(x), sin(x)*∂(x) Here, sin(x) should be shifted to the first argument, This function does that job
"""
function merge_args∫(expr)
    if typeof(operation(expr))!=typeof(∫) return nothing end 

    arg1, arg2 = arguments(expr)

    if typeof(operation(arg2)) in [typeof(*),typeof(/)]
        u = occursin_operator(∂, arg2)

        if isnothing(u) return nothing end

        # Make a function to get ∂ and other terms from arg2, and then just multiply the other term with arg1 to get terms in correct positions 
        v = arg2 / u
        arg1 = *(arg1, v)

        return ∫(arg1, u)
    else
        return nothing 
    end
end

# tree[1][2]
# e1 = Symbolics.value(∫(sqrt(9 - (sin(x)^2)) / (9sin(x)^2), 3cos(x)*∂(x)))
# merge_args∫(e1)

# ∫(sqrt(9 - (sin(x)^2)) / (9(sin(x)^2)), 3cos(x)*∂(x)) 
# ∫((cos(x)*sqrt(9 - (9//1)*(sin(x)^2))) / (cos(x)*sqrt(9 - (9//1)*(sin(x)^2))), ∂(x))
# occursin_operator(∂, e1)
# arg1, arg2 = arguments(e1)


# a = [sin(x), cos(x), ∂(x)]
# any(e -> isequal(e, ∂(x)), a)
# any(e -> isequal(e, log(x)), a)
# const_out(∫(sin(y)*2*cos(x)*log(x), ∂(x)))

# int_rules[end](∫(log(2y), ∂(2y)))

"""
Performs the operation: ∫(u*v)dx = u ∫v*dx - ∫ (∂u/∂x) * (∫v*dx) * dx
    
Parameters:
    u: first function to be taken in by-parts (one which is to be differentiated)
    v: all the other functions except u will be taken as v
"""
function by_parts(expr, u=nothing)
    byparts_applicable = @rule ∫(~uv, ∂(~x)) => true

    # if it's not in compatible form, or it u itself is not in integral
    if byparts_applicable(expr)===nothing || (!isnothing(u) && !any(isequal(u), arguments(arguments(expr)[1])))
        return nothing
    end

    if (u===nothing)
        if typeof(operation(arguments(expr)[1])) == typeof(*)
            u = arguments(arguments(expr)[1])[1]  # just choose the first function as u 
        else  # if the form is not ∫(u*v, ∂(x)), then select like ∫(u, ∂(x))
            u = arguments(expr)[1]
        end
    end

    if iscall(u) && typeof(operation(u)) == typeof(*)
        u = arguments(u)
        if issubset(Set(u), Set(arguments(arguments(expr)[1])))
            v = reduce(*,setdiff(Set(arguments(arguments(expr)[1])), Set(u)))
            u = reduce(*, u)
        end
    elseif typeof(operation(arguments(expr)[1])) == typeof(*)
        v = reduce(*,setdiff(Set(arguments(arguments(expr)[1])), Set([u])))
    else
        v = 1
    end

    dx = arguments(expr)[2]

    return -(
        u*∫(v, dx),
        ∫((∂(u)/dx) * ∫(v, dx), dx)
    )
    
end

# xmx = by_parts(Symbolics.value(∫(sin(2t), ∂(t))))
# typeof(xmx)


# I think this should be left on the experienced arm of substituion model 
function substitute∫(expr::SymbolicUtils.BasicSymbolic{Real}, subs::Dict{SymbolicUtils.BasicSymbolic{Real}, SymbolicUtils.BasicSymbolic{Real}})
    if typeof(operation(expr))!=typeof(∫) return nothing end

    # Whenever we do substituion there is one type of variable this side and other should be on that side
    # Handle that thing

    arg1, arg2 = arguments(expr)

    # First substitute the argument
    substitute(arg1, subs)

    # Then substitute the derivative
    der1 = apply_∂(Symbolics.value(∂(first(subs)[1])))
    der2 = apply_∂(Symbolics.value(∂(first(subs)[2])), Symbolics.value(t))

    # Then process the terms that haven't been substituted 

end

########################## Differentiation Rules ###############################
"""
Apply Differentiation wrt parameter 'x'
"""
function apply_∂(expr::SymbolicUtils.BasicSymbolic{Real}, x::SymbolicUtils.BasicSymbolic{Real}=Symbolics.value(x))
    dx = Differential(x)

    if typeof(operation(expr)) != typeof(∂)
        return nothing
    end

    return *(expand_derivatives(dx(arguments(expr)[1])), ∂(x))
end
# apply_∂(Symbolics.value(∂(exp(x)*x^2)), Symbolics.value(x))


"""
Differentiate tuple, and returns tuple of derivative

Arguments:
    expr = Expression that has to be differentiated in form of Vectors of tuple
    wrt = Differentiation with respect to this variable, it's form should be same as that of expr
"""
function apply_∂(
    expr::Vector{Vector{SymbolicUtils.BasicSymbolic{Real}}},
    wrt::Vector{Vector{SymbolicUtils.BasicSymbolic{Real}}})

    @assert length(expr)==length(wrt)  "Length of expr and wrt should be same"

    ans = Vector{Vector{SymbolicUtils.BasicSymbolic{Real}}}()
    for i in 1:length(expr)
        expr_i = Vector{SymbolicUtils.BasicSymbolic{Real}}()
        for j in 1:length(expr[i])
            push!(expr_i, apply_∂(∂(expr[i][j]), wrt[i][j]))
        end

        push!(ans, expr_i)
    end

    return ans
end

# subsi = Vector{Vector{SymbolicUtils.BasicSymbolic{Real}}}()
# wrt = [[Symbolics.value(t), Symbolics.value(x)]]
# push!(subsi, [Symbolics.value(exp(t)), Symbolics.value(3sec(x))])
# apply_∂(subsi, wrt)

################################### Too much basic rules ##############################

"""
To take common out of an expression
"""
function take_common(expr, common)
    if typeof(operation(expr))!=typeof(+) return nothing end 

    u = Vector{Any}()

    for arg in arguments(expr)
        push!(u, arg / common)
    end

    return (reduce(+, u)) * common
end

# ex =Symbolics.value( x + sin(x) + x*tan(x))
# take_common(ex, x)

# Convert this to any arbitarary power
function take_power(expr, power)
    if iscall(expr)
        if typeof(operation(expr)) == typeof(+) return nothing end

        if typeof(operation(expr)) == typeof(*)  # x^2 * sin(x) * tan(x)^4
            u = Vector{Any}()
            for arg in arguments(expr)
                push!(u, arg^(power))
            end

            return reduce(*, u)
        elseif typeof(operation(expr)) == typeof(^)  # [x*sin(x)]^2
            return ^(arguments(expr)[1], arguments(expr)[2]*power)
        end
        
    else
        return nothing
    end
end

function take_sqrt(expr)
    if typeof(operation(expr)) != typeof(sqrt) return nothing end

    u = Vector{Any}()

    for arg in arguments(expr)
        push!(u, take_power(arg, 1/2))
    end

    return reduce(*, u)
end

# ex1 = Symbolics.value(sin(x) - cot(x) - tan(x))
# take_power(ex1,1/2)

# ex1 = Symbolics.value(sqrt(x*sin(x)^2))
# take_sqrt(ex1)

# ex1 = Symbolics.value(x*sin(x)^2)
# take_power(ex1,1/2)


"""
    complete_square(expr, x)

Given a quadratic expression `expr` in variable `x`, return its completed square form.
"""
function complete_square(expr)
    # Extract coefficients
    a = Symbolics.coeff(expr, x^2)
    b = Symbolics.coeff(expr, x)
    c = substitute(expr, x => 0)

    # Compute the completed square
    h = b / (2a)
    k = h^2 - c
    completed_form = a * (x + h)^2 - a*h^2 + c

    return simplify(completed_form)
end

# expr = 3x^2 - 12x + 7
# expr = x^2 + 4x - 5
# completed = complete_square(expr, x)

"""
Find quadratic roots

Returns nothing if the roots aren't real
"""
function find_quad_roots(expr, x=x)
    a = Symbolics.coeff(expr, x^2)
    b = Symbolics.coeff(expr, x)
    c = substitute(expr, x => 0)

    discriminant = b^2 - 4a*c

    if discriminant<0
        return nothing
    end

    root1 = (-b + sqrt(discriminant)) / (2a)
    root2 = (-b - sqrt(discriminant)) / (2a)

    return root1, root2
end

# ex = Symbolics.value(2 + 3 * x + x^2)
# find_quad_roots(ex)

# Symbolics.degree(ex, x)
# Symbolics.get_variables(ex)

"""
Find roots and then simply write (x-root1)*(x-root2)
"""
function mid_term_split(expr, var=Symbolics.value(x))
    @assert Symbolics.degree(ex, var)>2 "Expression should be quadratic"

    root1, root2 = find_quad_roots(expr, var)
    return (var-root1)*(var-root2)
end

"""
Does partial fraction decomposition

It will have to involve substituion model 
"""
function partial_fraction(expr)
    if !iscall(expr) return nothing end
    if typeof(operation(expr))==typeof(/)


    end
end

# Checks whether a rule can be applied on some expression or not
function check_rules(ex, rules)
    applicable = []
    for r in rules

        #Storing the rules are beneficial here, as computing them can be costly depending on the rule 
        r_ex = r(ex)
        if r_ex !== nothing
            push!(applicable, (r, r_ex))
        end
    end

    applicable
end