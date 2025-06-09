# Try to make the integration function ∫ symbolic, so that by part rule can be applied on it
# And it can also call the model inside that function

# Or should we just utilize the ∫ symbol as symbolic, and should manipulate the ∫ from model 
# And from the expression tree should see what operator is it, and what differential does it have

@register_symbolic ∫(expr, dx)  # making of integration operator
@register_symbolic ∂(expr)  # differential opearator

# Define ∫ and ∂ for special cases
function ∫(::Number, dx)
    return ∫(1, dx)
end

function ∂(a::Number)
    return 0
end

"""
A tool from which model can take any action (func) over (node) of an (expression tree)

It is combination of three functions:
    1. Taking Action at some node 
    2. Modifying the expression tree 
    3. Traversing the modified tree 

Returns:
    New modified expression tree 
"""
function take_action(expr_tree, func, node; args=[])
    xmx = Symbolics.value(func(expr_tree[node+1][2], args...))  # Taking Action at some node 
    if !(xmx === nothing)
        xmx = substitute(expr_tree[1][2], Dict(expr_tree[node+1][2] => xmx), fold=false)   # Modifying the expression tree 
        return traverse_expr(Symbolics.value(xmx), returnTree=true)
    else
        return nothing
    end
end

# ex = ∫(3*x, ∂(x))
# ex = ∫(sin(x) * x, ∂(x))
# e1 = Symbolics.value(ex)

# if typeof(operation(arguments(e1)[1])) == typeof(∂)
#     print("yesy")
# end
# typeof(operation(e1))

function big_thoughts()
    """
    ∫(ex, dx)
    
    Computes integral of ex with respect to dx
    
    Once we have an expression check what are the actions that can be taken on that expression
    Take some expression and see what happens
    If the integral comes in the form of known integral then we are done
    
    This func can call more than one model at one ∫
    """

    function ∫(ex::SymbolicUtils.BasicSymbolic{Number}, dx::Differential)
        return dx(ex)
    end

    """
    Model takes action, traverse in the tree to bring ∫ in the standar form 
    
    Actions that model can take: 
        Model should be able to go ahead in the tree apply some rules and come back in the tree. 
        Take some substitution
    """
    function ∫Model(ex::SymbolicUtils.BasicSymbolic{Number}, dx::Differential)

        # First check all the rules that are applicable in this expression
        rules = check_rules(rules, trigs_rules)

        op = operation(ex)
        arg = arguments(ex)

    end
end

# Recursive function to traverse the expression tree
function traverse_expr(expr, tree=nothing; parent::Int=-1, current::Int=-1, returnTree::Bool=false, returnTreeForPlot::Bool=false, printTree::Bool=false)
    current += 1
    if SymbolicUtils.iscall(expr)
        arg, op = arguments(expr), operation(expr)

        (printTree) && println("Operation: $op  Parent: $parent  Current: $current  Expression: $expr")

        # Store the tree
        if (returnTreeForPlot)
            push!(tree, (parent, current, op))
        elseif (returnTree)
            push!(tree, (current, expr))
        end

        parent = current

        for a in arg
            current = traverse_expr(a, tree, parent=parent, current=current, returnTree=returnTree, returnTreeForPlot=returnTreeForPlot, printTree=printTree)
        end

        return current
    else
        (printTree) && println("Operation: $expr  Parent: $parent  Current: $current  Expression: $expr")

        if (returnTreeForPlot)
            push!(tree, (parent, current, expr))
        elseif (returnTree)
            push!(tree, (current, expr))
        end

        return current
    end
end

"""Closure for traverse_expr"""
function traverse_expr(expr; parent::Int=-1, current::Int=-1, returnTree::Bool=false, returnTreeForPlot::Bool=false, printTree::Bool=false)
    if (returnTree)
        if (returnTreeForPlot)
            tree = Vector{Tuple{Int64,Int64,Any}}()
        else
            tree = Vector{Tuple{Int64,Any}}()
        end
        traverse_expr(expr, tree, parent=parent, current=current, returnTree=returnTree, returnTreeForPlot=returnTreeForPlot, printTree=printTree)
        return tree
    else
        traverse_expr(expr, nothing, parent=parent, current=current, returnTree=returnTree, returnTreeForPlot=returnTreeForPlot, printTree=printTree)
    end
end

"""
To traverse equations

Made basically to be used for dataset
"""
function traverse_expr(expr::Equation; returnTreeForPlot::Bool=false)
    lhs = traverse_expr(expr.lhs, returnTree=true, returnTreeForPlot=returnTreeForPlot)
    rhs = traverse_expr(expr.rhs, returnTree=true, returnTreeForPlot=returnTreeForPlot)

    lhs = Vector{Tuple{Int64,Int64,String}}([(a + 1, b + 1, string(c)) for (a, b, c) in lhs])
    rhs = Vector{Tuple{Int64,Int64,String}}([(a + 1 + length(lhs), b + 1 + length(lhs), string(c)) for (a, b, c) in rhs])
    rhs[1] = (0, rhs[1][2], string(rhs[1][3]))
    push!(lhs, (-1, 0, string(~)))
    push!(lhs, rhs...)

    sort!(lhs, by=x -> x[2])

    return lhs
end

"""To print all the elements of any iteration"""
function show_full(iter::Any)
    for e in iter
        println(e)
    end
end


# expr = Symbolics.value((cos(x)-tan(x))~sin(x)+cot(x))
# tree = traverse_expr(expr, returnTreeForPlot=true)

# Traverse the example expression
# using BenchmarkTools

# ex = Symbolics.value(sin(x) * cos(x) + tan(x))
# traverse_expr(ex, returnTreeForPlot=true)

# tree = Vector{Tuple{Int,Int,Any}}()

# tree = traverse_expr(ex, returnTree=true, returnTreeForPlot=true)
# show_full(tree)

# ex = Symbolics.value(/(-(cot(x), tan(x)), +(1, cos(*(4, x)))))
# println("Traversing expression: ", ex)

# tree = Vector{Tuple{Int,Int,Any}}()
# tree = Vector{Tuple{Int,Any}}()

# @btime traverse_expr(ex, returnTree=true, tree=tree, printTree=false)  # printTree
# traverse_expr(ex, returnTree=true, tree=tree, printTree=false)  # printTree
# traverse_expr(ex, printTree=true)  # printTree


# One of the idea is that we will traverse the whole tree beforehand and will check all the possible actions that can be taken, and then the model will decide which action to take, as taking action only change one or two nodes in which action is taking place
# and because of this model will have a bigger picture of what is happening around it
