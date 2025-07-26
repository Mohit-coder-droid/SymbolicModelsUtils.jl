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
        expr_tree[side][2][node] += -1 * expr_tree[side][2][node]
    else
        if (!isequal(expr_tree[side][2][node], x))
            if expr_tree[side][1] == *
                other_side /= expr_tree[side][2][node]
                expr_tree[side][2][node] /= expr_tree[side][2][node]
            elseif expr_tree[side][1] == /
                other_side *= expr_tree[side][2][node]
                expr_tree[side][2][node] *= expr_tree[side][2][node]
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

    lhs[1](lhs[2]...) ~ rhs[1](rhs[2]...)
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

# que12 = 

# There is something wrong with linear_transport it doesn't able to handle 3 or more arguments in an operation
# xmx = 0 ~ 1 + -2 / x
# linear_transport(deepcopy(xmx), 2, 2)   # 2 / x ~ 1 + 2 / x + -2 / x  => because julia is not using it's brain that it can be simplified very easily
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
