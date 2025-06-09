

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
    other_side = expr_tree[3-side][1](expr_tree[3-side][2]...)
    if (expr_tree[side][1] == +)
        other_side = other_side + -1 * expr_tree[side][2][node] # adding to any of the node will bring the same result
        expr_tree[side][2][node] += -1 * expr_tree[side][2][node]
    elseif expr_tree[side][1] == *
        other_side /= expr_tree[side][2][node]
        expr_tree[side][2][node] /= expr_tree[side][2][node]
    elseif expr_tree[side][1] == /
        other_side *= expr_tree[side][2][node]
        expr_tree[side][2][node] *= expr_tree[side][2][node]
    else
        error("Equation should be linear")
    end

    if side == 1
        return [linear_tree(expr_tree[1][1](expr_tree[1][2]...)), linear_tree(other_side)]
    else
        return [linear_tree(other_side), linear_tree(expr_tree[2][1](expr_tree[2][2]...))]
    end
end

function linear_transport(expr::Equation, side::Int=1, node::Int=1)
    tree = [linear_tree(expr.lhs), linear_tree(expr.rhs)]
    if side == 1
        lhs, rhs = linear_transport(tree, side, node)
    elseif side == 2
        lhs, rhs = linear_transport(tree, side, node)
    else
        error("Side can be either 1 or 2, that means either LHS or RHS")
    end

    lhs[1](lhs[2]...) ~ rhs[1](rhs[2]...)
end

# que12 = 

# There is something wrong with linear_transport it doesn't able to handle 3 or more arguments in an operation
# xmx = linear_transport(deepcopy(yup[4]), 2, 1)
# linear_transport(deepcopy(xmx), 1, 3)
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
