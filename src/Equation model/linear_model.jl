# yup = linear_eq(10, 4)
# que = yup[1]

# operation(yup[1].lhs)
# arguments(yup[1].lhs)
function linear_tree(expr)
    if iscall(expr)
        (operation(expr), arguments(expr))
    else
        (+, Any[expr])
    end
end

function linear_transport(expr_tree, side::Int=1, node::Int=1)
    if length(expr_tree[side][2]) < node
        return expr_tree  # return the tree without affecting when actions are not applicable 
    end
    other_side = expr_tree[3-side][1](expr_tree[3-side][2]...)
    if (expr_tree[side][1] == +)
        other_side = other_side + -1 * expr_tree[side][2][node] # adding to any of the node will bring the same result
        expr_tree[side][2][node] = 0
    else
        if (!occursin(Symbolics.value(x), expr_tree[side][2][node]))  # any action that is making x in denominator will be flagged as inappropriate action 
            if expr_tree[side][1] == *
                other_side /= expr_tree[side][2][node]
                expr_tree[side][2][node] = 1
            elseif expr_tree[side][1] == /
                other_side *= expr_tree[side][2][node]
                expr_tree[side][2][node] = 1
            else
                error("Equation should be linear")
            end
        else
            return expr_tree
        end
    end

    if side == 1
        return [linear_tree(expr_tree[1][1](expr_tree[1][2]...)), linear_tree(other_side)]
    else
        return [linear_tree(other_side), linear_tree(expr_tree[2][1](expr_tree[2][2]...))]
    end
end

"""
    linear_transport(expr::Equation, side::Int=1, node::Int=1)

    Transport a term from one side to another in a linear equation

    # Arguments
    - `expr`: Equation
    - `side::Int`: 1 for LHS and 2 for RHS 
    - `node::Int`: 1 for first term and 2 for second term 

    # Returns 
    It return a new expression tree, in which the term is transferred to the other side
"""
function linear_transport(expr::Equation, side::Int=1, node::Int=1)
    tree = deepcopy([linear_tree(expr.lhs), linear_tree(expr.rhs)])
    if side == 1
        lhs, rhs = linear_transport(tree, side, node)
    elseif side == 2
        lhs, rhs = linear_transport(tree, side, node)
    else
        error("Side can be either 1 or 2, that means either LHS or RHS")
    end

    # in linear, I think the only work of simplify, is to make things to take common, if they are having same coefficient
    take_common(Symbolics.value(lhs[1](lhs[2]...)), Symbolics.value(x), false) ~ take_common(Symbolics.value(rhs[1](rhs[2]...)), Symbolics.value(x), false)
end

"""
    linear_termination_status(expr::Equation)
    
Checks whether a linear equation is in it's termination state or not

Termination state for linear equation is defined as: x ~ const
"""
function linear_termination_status(expr::Equation)
    if isequal(expr.lhs, x) & !any(v -> isequal(v, x), Symbolics.get_variables(expr.rhs))
        return true
    elseif isequal(expr.rhs, x) & !any(v -> isequal(v, x), Symbolics.get_variables(expr.lhs))
        return true
    else
        false
    end
end

# module linear_system_module
# using Symbolics, SymbolicUtils
mutable struct linear_system
    eqs::Vector{Equation}
    vars::Vector{Num}
    coeffs::Matrix{Num}  # fake coeff just for calculation
    real_coeffs::Matrix{Num}  # coeff what model will see and the real coeff

    num_eqs::Int
    num_vars::Int
end
# end

# Symbolics.coeff(eqs.eqs[1].lhs, eqs.vars[2])

# Check whether the equations are solvable (leading symbolic coefficients)
"""
    generate_system(num_eqs::Int, num_vars::Int)

    Generate a system of `num_eqs` linear equations with `num_vars` variables.
    - Variables: x₁, x₂, …, xₙ
    - Coefficients: aᵢⱼ (for row i, column j)
    - RHS constants: dᵢ

    # Examples:
    ```julia-repl 
    julia> eqs, vars, coeffs, rhs = linear_system(3,3)

    julia> eqs
    3-element Vector{Equation}:
    a_1_1*x_1 + a_1_2*x_2 + a_1_3*x_3 ~ b_1
    a_2_1*x_1 + a_2_2*x_2 + a_2_3*x_3 ~ b_2
    a_3_1*x_1 + a_3_2*x_2 + a_3_3*x_3 ~ b_3
    ```

    Returns a vector of equations.
"""
function linear_system(num_eqs::Int=1, num_vars::Int=1; first_time::Bool=true, coeffs_::Union{Nothing,Matrix{Num}}=nothing)::linear_system
    # Create variables x₁, x₂, ..., xₙ and coefficients aᵢⱼ and RHS dᵢ
    vars = [Symbolics.variable(Symbol("x_$j")) for j in 1:num_vars]
    coeffs = [Symbolics.variable(Symbol("p_$(i)_$(j)")) for i in 1:num_eqs, j in 1:num_vars]
    rhs = [Symbolics.variable(Symbol("p_$(i)_4")) for i in 1:num_eqs]

    # when initializing a linear eq, coeff and real coeff are same
    if first_time
        coeffs_ = [Symbolics.variable(Symbol("a_$(i)_$(j)")) for i in 1:num_eqs, j in 1:num_vars]
        rhs_ = [Symbolics.variable(Symbol("a_$(i)_4")) for i in 1:num_eqs]

        coeffs_ = hcat(coeffs_, rhs_)
    end

    # Build equations
    eqs = [sum(coeffs[i, j] * vars[j] for j in 1:num_vars) ~ rhs[i] for i in 1:num_eqs]

    return linear_system(eqs, vars, hcat(coeffs, rhs), coeffs_, num_eqs, num_vars)
end

function show_real_linear_system(eqs::linear_system)
    return [sum(eqs.real_coeffs[i, j] * eqs.vars[j] for j in 1:eqs.num_vars) ~ eqs.real_coeffs[i, 4] for i in 1:eqs.num_eqs]
end

"""
Gives the general form of the linear equation by introducing new variables, and keeping the relation of the old one, so that it doesn't become too difficult to model to take action 

What model will see that it's transferring coefficients of variables, but at the background something totally different is happening

We can keep track of all the relation coefficients 
"""
function linear_system(eqs::linear_system)
    coeffs_lhs = [Symbolics.coeff(eqs.eqs[i].lhs, eqs.vars[j]) for i in 1:length(eqs.eqs), j in 1:length(eqs.vars)]

    coeffs_rhs = [Symbolics.coeff(eqs.eqs[i].rhs, eqs.vars[j]) for i in 1:length(eqs.eqs), j in 1:length(eqs.vars)]

    const_rhs = Vector{Num}([])
    const_lhs = Vector{Num}([])
    for i in 1:length(eqs.eqs)
        # some lhs or rhs aren't callable 

        # Simple way to find the constant term (just subtract the variable terms)
        if iscall(eqs.eqs[i].lhs) && (operation(eqs.eqs[i].lhs) in [+, -])
            push!(const_lhs, simplify(eqs.eqs[i].lhs - operation(eqs.eqs[i].lhs)(sum(coeffs_lhs[i, :] .* eqs.vars))))
        end
        if iscall(eqs.eqs[i].rhs) && (operation(eqs.eqs[i].rhs) in [+, -])
            push!(const_rhs, simplify(eqs.eqs[i].rhs - operation(eqs.eqs[i].rhs)(sum(coeffs_rhs[i, :] .* eqs.vars))))
        end

        # If there is multiplication, i.e., there is just one term in that case just check whether that term is variable or const 
        if !iscall(eqs.eqs[i].lhs) || (operation(eqs.eqs[i].lhs) in [*, /])
            if isempty(intersect(Set(Symbolics.get_variables(eqs.eqs[i].lhs)), Set(eqs.vars)))
                push!(const_lhs, eqs.eqs[i].lhs)
            else
                push!(const_lhs, 0)
            end

            # if any(occursin(eqs.vars[j], eqs.eqs[i].lhs) for j in 1:length(eqs.vars))
            #     push!(const_lhs, 0)
            # else
            #     push!(const_lhs, eqs.eqs[i].lhs)
            # end
        end

        if !iscall(eqs.eqs[i].rhs) || (operation(eqs.eqs[i].rhs) in [*, /])
            if isempty(intersect(Set(Symbolics.get_variables(eqs.eqs[i].rhs)), Set(eqs.vars)))
                push!(const_rhs, eqs.eqs[i].rhs)
            else
                push!(const_rhs, 0)
            end

            # if any(occursin(eqs.vars[j], eqs.eqs[i].rhs) for j in 1:length(eqs.vars))
            #     push!(const_rhs, 0)
            # else
            #     push!(const_rhs, eqs.eqs[i].rhs)
            # end
        end
    end

    # this relation are with respect to new variables
    coeffs = Num.(coeffs_lhs - coeffs_rhs)
    rhs = Num.(const_rhs - const_lhs)

    # Create relation w.r.t old ones, go to every coeff, get all the coeff, and replace it with old ones, and update the old coeff 
    function get_indices(term_)
        parts = split(string(term_), "_")
        return parse.(Int, parts[2:end])
    end

    coeffs = hcat(coeffs, rhs)
    for i in eachindex(coeffs)
        coeff = coeffs[i]
        for term_ in Symbolics.get_variables(coeff)
            subsi_term = eqs.real_coeffs[get_indices(term_)...]
            coeff = substitute(coeff, Dict([term_ => subsi_term]))
        end
        coeffs[i] = simplify(coeff)
    end

    return linear_system(eqs.num_eqs, eqs.num_vars; first_time=false, coeffs_=coeffs)
end

struct linear_system_action
    do_combine::Bool
    # If trying to combine two equation
    eq1::Int
    eq2::Int  # eq to be affected 
    combine_type::Int  # [add, subtract]

    do_single::Bool  # if both do_combine and do_single are true, then do_single will be tackled first
    # If manipulating a single eq
    eq_nu::Int  # which equation to be manipulated
    action::Int  # [1,2,3,4]  [Add, Subtract, mult, divide] any term in that eq

    # From where this term will going to come, can I provide a separate matrix of coefficients 
    term_loc::Tuple{Int,Int}
end

function linear_system_action(;
    do_combine::Bool=false,
    eq1::Int=1,
    eq2::Int=2,
    combine_type::Int=1,
    do_single::Bool=false,
    eq_nu::Int=1,
    action::Int=1,
    term_loc::Tuple{Int,Int}=(1, 1)
)
    return linear_system_action(do_combine, eq1, eq2, combine_type, do_single, eq_nu, action, term_loc)
end

function linear_system_transport(eqs::linear_system, action::linear_system_action)
    eqs = deepcopy(eqs)
    function modify_eq(eq, term_, op)
        if op in [+, -]
            return op(eq.lhs, term_) ~ op(eq.rhs, term_)
        elseif op in [*, /]   # do this operations 
            if iscall(eq.lhs)
                op_eq_lhs = operation(eq.lhs)
                if op_eq_lhs in [+, -]
                    if op == *
                        lhs = op_eq_lhs(arguments(eq.lhs) .* term_...)
                    else
                        lhs = op_eq_lhs(arguments(eq.lhs) ./ term_...)
                    end
                else
                    lhs = op(eq.lhs, term_)
                end
            else
                lhs = op(eq.lhs, term_)
            end

            if iscall(eq.rhs)
                op_eq_rhs = operation(eq.rhs)
                if op_eq_rhs in [+, -]
                    if op == *
                        rhs = op_eq_rhs(arguments(eq.rhs) .* term_...)
                    else
                        rhs = op_eq_rhs(arguments(eq.rhs) ./ term_...)
                    end
                else
                    rhs = op(eq.rhs, term_)
                end
            else
                rhs = op(eq.rhs, term_)
            end

            return lhs ~ rhs
        end
    end

    function combine_eq(eq1, eq2, op)
        return op(eq1.lhs, eq2.lhs) ~ op(eq1.rhs, eq2.rhs)
    end

    if action.do_single
        if action.action == 1
            eqs.eqs[action.eq_nu] = modify_eq(eqs.eqs[action.eq_nu], eqs.coeffs[action.term_loc...], +)
        elseif action.action == 2
            eqs.eqs[action.eq_nu] = modify_eq(eqs.eqs[action.eq_nu], eqs.coeffs[action.term_loc...], -)
        elseif action.action == 3
            eqs.eqs[action.eq_nu] = modify_eq(eqs.eqs[action.eq_nu], eqs.coeffs[action.term_loc...], *)
        elseif action.action == 4
            if !isequal(eqs.real_coeffs[action.term_loc...], 0)
                eqs.eqs[action.eq_nu] = modify_eq(eqs.eqs[action.eq_nu], eqs.coeffs[action.term_loc...], /)
            end
        end
    end

    if action.do_combine
        if action.combine_type == 1
            eqs.eqs[action.eq2] = combine_eq(eqs.eqs[action.eq2], eqs.eqs[action.eq1], +)
        elseif action.combine_type == 2
            eqs.eqs[action.eq2] = combine_eq(eqs.eqs[action.eq2], eqs.eqs[action.eq1], -)
        end
    end

    return linear_system(eqs)
end

"""
    Checks whether the linear system has reached it's termination state or not
"""
function linear_system_termination(eqs::linear_system)::Bool
    real_eqs = show_real_linear_system(eqs)

    for eq in real_eqs
        if any(v -> isequal(v, eq.lhs), eqs.vars)
            continue
        end
        if any(v -> isequal(v, eq.rhs), eqs.vars)
            continue
        end
        return false
    end
    return true
end

# linear_system_termination(eqs6)

# for i in 1:3, j in 1:4
# name = Symbol("a_$(i)_$(j)")
#     eval(:(@variables $name))
#     end
# @variables x_1 x_2 x_3

# eq = (-a_3_1 - 3(a_2_1 + (a_1_1 + a_2_1)*a_2_2)*a_3_3)*x_1 + (-a_3_2 + 3(-a_2_2 - a_1_2*a_2_2 - (a_2_2^2))*a_3_3)*x_2 + (-a_3_3 + 3(-a_2_3 - a_1_3*a_2_2 - a_2_2*a_2_3)*a_3_3)*x_3 ~ -a_3_4 - 3(a_2_4 - a_1_1*a_2_1 + (a_1_4 + a_2_4)*a_2_2 - a_2_3*a_3_1 - (a_1_1^2)*a_2_2 + a_1_1*(a_2_1 + a_1_1*a_2_2) + (2a_2_3 + 2a_1_3*a_2_2 + a_2_2*a_2_3)*a_3_1 + (-a_2_3 - a_1_3*a_2_2)*a_3_1 - a_1_3*a_2_2*a_3_1 - a_2_2*a_2_3*a_3_1)*a_3_3 + 2(a_3_1 + (a_2_1 + (a_1_1 + a_2_1)*a_2_2)*a_3_3)*(-a_2_3 - a_1_3*a_2_2 - a_2_2*a_2_3)*a_3_3 + 2(-a_3_1 - (a_2_1 + (a_1_1 + a_2_1)*a_2_2)*a_3_3)*(-a_2_3 - a_1_3*a_2_2 - a_2_2*a_2_3)*a_3_3

# tree = traverse_expr(eq)  # length>100, then how is this coming up 
# maybe this is the problem of env_checker not of my environment, as I had implmeneted that length check logic in the model not the environment

# make everything 0
# eqs = linear_system(3, 3)
# linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=2, eq1=3, eq2=3))
# linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=2, eq1=1, eq2=1))
# eqs_ = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=2, eq1=2, eq2=2))

# eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(3, 3), eq_nu=3, eq1=2, eq2=1))
# eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(3, 1), eq_nu=3, eq1=2, eq2=1))
# eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=1, action=4, term_loc=(3, 1), eq_nu=3, eq1=3, eq2=1))
# show_real_linear_system(linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=1, action=4, term_loc=(3, 1), eq_nu=3, eq1=3, eq2=1)))

function solution()
    # 1. The whole linear equation can be solved using a simple strategy that to remove a variable, first divide it by it's own coefficients, and then multiply it by the coefficient from the variable which has to be removed. And then subtract those two eq
    # Strategy for solving linear eq 
    #   455.163 ms (10242105 allocations: 584.19 MiB)

    # After applying deepcopy
    # 460.219 ms (10779373 allocations: 589.45 MiB)
    begin
        # @btime begin
        eqs = linear_system(3, 3)
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(2, 3), eq_nu=2, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(1, 3), eq_nu=2, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=2, eq1=2, eq2=1))
        # eqs_ = linear_system(eqs)

        # Removing the same variable from eq3
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(2, 3), eq_nu=2, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(3, 3), eq_nu=2, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=4, term_loc=(1, 3), eq_nu=2, eq1=2, eq2=3))
        # eqs1 = linear_system(eqs)

        # Removing another variable from eq2
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(1, 2), eq_nu=1, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(3, 2), eq_nu=1, eq1=2, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=4, term_loc=(1, 3), eq_nu=2, eq1=1, eq2=3))
        # eqs2 = linear_system(eqs1)

        # Instead of calling some other model for solution, the soln can be easily achieved using actions of here also
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(3, 1), eq_nu=3, eq1=1, eq2=3))
        # eqs3 = linear_system(eqs2)  # we got x_1

        # To get x_2, remove x_1 from eq1 
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=3, eq1=1, eq2=3))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(1, 1), eq_nu=3, eq1=3, eq2=1))
        # end

        # get the soln back 
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(3, 1), eq_nu=3, eq1=1, eq2=3))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(1, 2), eq_nu=1, eq1=3, eq2=1))
        # eqs4 = linear_system(eqs3)

        # Get the last variable 
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(2, 2), eq_nu=1, eq1=3, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=3, term_loc=(2, 1), eq_nu=3, eq1=3, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(2, 1), eq_nu=3, eq1=3, eq2=2))
        eqs = linear_system_transport(eqs, linear_system_action(do_combine=true, combine_type=2, action=3, term_loc=(2, 1), eq_nu=3, eq1=1, eq2=2))
        # eqs5 = linear_system(eqs4)

        # Get the soln 
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(1, 2), eq_nu=1, eq1=3, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(2, 3), eq_nu=2, eq1=3, eq2=1))
        eqs = linear_system_transport(eqs, linear_system_action(do_single=true, combine_type=2, action=4, term_loc=(3, 1), eq_nu=3, eq1=3, eq2=1))

        linear_system_termination(eqs)
    end
    # eqs6 = linear_system(eqs5)

    # lhs, rhs = linear_system(eqs)
end

# [1 1 1 1 1 0 0 1 3 0]  => this action makes the terms 0
# trr = traverse_expr(0 ~ 0, returnTreeForPlot=true)

# yup = [Symbolics.coeff(eqs.eqs[3].lhs, vars[i]) for i in 1:length(vars)]

# simplify(eqs.eqs[2])

# Mention this that any action that is making x in denominator will be flagged as inappropriate action 
# que12 = 

# There is something wrong with linear_transport it doesn't able to handle 3 or more arguments in an operation
# expr1 = (66x - (39 // 1) * y) / z ~ 120

# for i in 1:1000
#     println(linear_eq(5))
# end
# expr = linear_eq(5)
# expr = -94x * y ~ (7 // 6) + 67x + 121z   # x*(-67 - 94y) ~ (7//6) + 121z => nothing ~ (7//6) + 121z  on linear_transport(deepcopy(expr), 1, 2)
# expr = 0 ~ x * (a + b * x)
# expr_tree = deepcopy([linear_tree(expr.lhs), linear_tree(expr.rhs)])
# expr = linear_transport(deepcopy(expr), 2, 1)   # 2 / x ~ 1 + 2 / x + -2 / x  => because julia is not using it's brain that it can be simplified very easily
# tree = traverse_expr(expr, returnTreeForPlot=true)
# linear_termination_status(expr)

# occursin(Symbolics.value(x), 1)

# x1 = Symbolics.values(x^2 + 3x)
# occursin(y, y^2 + 2)


# expr =32 + 5z ~ 4x - x*y   # In this case we have to add another action, and that will be to take common in between them 





# xmx = linear_transport(deepcopy(xmx), 2, 1)
# xmx = linear_transport(deepcopy(xmx), 2, 2)
# xmx = linear_transport(deepcopy(xmx), 2, 1)
# tre = linear_transport(tre, 1, 1)
# linear_transport(tre, 1, 2)
# tre = linear_transport(tre, 2, 1)
# xmx = linear_transport(tre, 2, 2)
# tre = [linear_tree(yup[2].lhs), linear_tree(yup[2].rhs)]
# tree[2][2][1]

# yupx = -45 + 73x ~ 106
# typeof(yupx.lhs)
# typeof(arguments(yupx.lhs)[1])
# arguments(yupx.lhs)

# ex = sin(x) * cot(x)
# arguments(Symbolics.value(ex))
