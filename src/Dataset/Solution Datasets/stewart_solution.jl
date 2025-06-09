include("../../integral.jl")
include("../../visualizer.jl")
include("../../Symbolics_func.jl")
include("../../Our_rules.jl")

@variables x y z t

function prob1()
    prob = [exp(x) * x^2, x, 3, 2 * exp(x) - 2 * exp(x) * x + exp(x) * x^2]

    # simplify is not working properly on this, maybe in future I can think of implementing some simplifying logic on my own
    simplify(2 * exp(x) - 2 * exp(x) * x + exp(x) * x^2)

    que = ∫(prob[1], ∂(x))

    # Actions taken by model
    que = Symbolics.value(que)

    # Firstly model should check whether this is integration problem or not, and integration wrt what variable (here it is x, i.e. dx)
    operation(que)
    dx = arguments(que)[2]  # check for derivative

    # check_rules(arguments(que)[1], trigs_rules)    # no need to check the rules for the whole expression according to my latest thinking

    que = by_parts(que, Symbolics.value(x^2))

    # Traverse the graph to know the current status and positions of the nodes, this will have to be done, to analyze what by-part has done to our expression tree
    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)

    # Model analyzed the expression and said that go to node 6, and take action of apply_∂, and change node 6 to whatever the result that we will get from it , and after that modify the tree accordingly
    xmx = apply_∂(tree[6+1][2], Symbolics.value(x))

    que = substitute(que, Dict(tree[6+1][2] => xmx), fold=false)

    # substitute will replace expression at all the places, but if I want to replace at some particular node then I think I will have to write my own function to replace on the basis of node number
    # which will be faster than substitute
    # I think I also have to implement some of my simplify rules to tackle some of the simplification of integrals and differentials
    # yup_ex = Symbolics.value(sin(4x) + 4x)
    # substitute(yup_ex, Dict(4x => y), fold=false)

    # Once again traverse tree 
    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)

    # Apply integration rule 1 on node 7 and then replace it with whatever we will get 
    xmx = int_rules[2](tree[7+1][2])
    que = substitute(que, Dict(tree[7+1][2] => xmx), fold=false)

    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)

    # There are three terms in *() so our by_part will not be able to catch it
    que = by_parts(tree[9+1][2], Symbolics.value(2x))

    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)
    xmx = apply_∂(tree[6+1][2], Symbolics.value(x))
    que = substitute(que, Dict(tree[6+1][2] => xmx), fold=false)


    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)

    # Apply the last integration rule
    xmx = int_rules[2](tree[16+1][2])
    que = substitute(que, Dict(tree[16+1][2] => xmx), fold=false)


    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)
    xmx = const_out(tree[8+1][2])
    que = substitute(que, Dict(tree[8+1][2] => xmx), fold=false)


    tree = Vector{Tuple{Int,Any}}()
    traverse_expr(que, returnTree=true, tree=tree, printTree=false)

    xmx = int_rules[2](tree[8+1][2])
    que = substitute(que, Dict(tree[8+1][2] => xmx), fold=false)  # the final answer!!! 
end

function prob2()
    prob = [exp(2 * x) * x, x, 2, -1 // 4 * exp(2 * x) + 1 // 2 * exp(2 * x) * x]
    que = Symbolics.value(∫(prob[1], ∂(x)))

    que = by_parts(que, Symbolics.value(x))

    # Make a substitution function to solve this question
    que = Symbolics.value(substitute(que, Dict(x => x / 2))) # here difficulty lies in model determing exactly (x=>x/2) and we also have to store all the substitutions that are being made to inverse them in the last in indefinite integration problem

    subsi = Vector{Dict{Num,Num}}()
    push!(subsi, Dict(x => 2x))  # inverse substitution as determined by the other model 

    # I think that the traversing operator will be implemented automatically as we will give the expression to our model, so we have to give position of nodes to it and to do that we have to traverse
    tree = traverse_expr(que, returnTree=true)

    # exp_tree = take_action(tree, apply_∂, 7; args=(Symbolics.value(x),))

    xmx = apply_∂(tree[7+1][2], Symbolics.value(x))
    que = substitute(que, Dict(tree[7+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = merge_args∫(tree[4+1][2])
    que = substitute(que, Dict(tree[4+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = const_out(tree[3+1][2])
    que = substitute(que, Dict(tree[3+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = int_rules[2](tree[6+1][2])
    que = substitute(que, Dict(tree[6+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = merge_args∫(tree[8+1][2])
    que = substitute(que, Dict(tree[8+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = const_out(tree[3+1][2])
    que = substitute(que, Dict(tree[3+1][2] => xmx), fold=false)

    tree = traverse_expr(que, returnTree=true)
    xmx = int_rules[2](tree[8+1][2])
    que = substitute(que, Dict(tree[8+1][2] => xmx), fold=false)


    # The question is not finished here, inverse the substitutions that had been done
    ans = substitute(que, pop!(subsi))

    # I feel merging actions of (taking action, modifying tree, and traversing tree) into 1 function, as they are keep coming again and again
end

function prob3()
    que = Symbolics.value(∫(t * cos(t) * sin(t), ∂(t)))
    tree = traverse_expr(que, returnTree=true)

    # multiply and divide by some expression
    tree[1+1][2] * 2
    trigs_rules[8](sin(2x))

    # Instead of multiplying and dividing 2 like that, I should give a force apply certain rule option, which will automatically tackle which thing to multiply and divide
    # This whole 3 operations should be one, when performed using substitution model 
    xmx = trigs_rules[8](tree[1+1][2] * 2) / 2  # substitution model
    que = substitute(que, Dict(tree[1+1][2] => xmx), fold=false)
    que = const_out(que)
    tree = traverse_expr(que, returnTree=true)

    # By parts
    tree = take_action(tree, by_parts, 2; args=(Symbolics.value(t),))
    que = tree[1][2]
    int_rules[5](tree[2+1][2])

    # Substituion process
    subsi = Vector{Dict{Num,Num}}()
    que = Symbolics.value(substitute(que, Dict(t => t / 2)))
    tree = traverse_expr(que, returnTree=true)
    push!(subsi, Dict(t => t / 2))  # inverse substitution as determined by the other model 

    # Combining two functions 
    tree = take_action(tree, apply_∂, 19, args=(Symbolics.value(t),))
    # que = tree[1][2]
    tree = take_action(tree, const_out ∘ merge_args∫, 6)
    # que = tree[1][2]

    tree = take_action(tree, const_out ∘ merge_args∫, 5)

    tree = take_action(tree, int_rules[6], 5)
    ans = tree[1][2]  # que is not finished here

    # Inverse the substitution, by substituion model 
    substitute∫(ans, pop!(subsi))
    ans = substitute(ans, Dict(t => 2t))

    # Actual answer: [t * cos(t) * sin(t), t, 3, -1 // 4 * t + 1 // 4 * cos(t) * sin(t) + 1 // 2 * t * sin(t)^2], which is right answer, just some simplification
    # If the model gives big answer, try to make it small, by some simplification 

end

function prob4()
    prob = [log(x)^2, x, 2, 2 * x - 2 * x * log(x) + x * log(x)^2]
    que = Symbolics.value(∫(prob[1], ∂(x)))
    # This problem can be solved by applying by parts after taking both u and v as log(x)
    # But let's take u = log(x)^2 and v = 1

    tree = traverse_expr(que, returnTree=true)
    tree = take_action(tree, by_parts, 0)

    tree = take_action(tree, const_out, 24)
    tree = take_action(tree, apply_∂, 6; args=(Symbolics.value(x),))

    tree = take_action(tree, const_out, 9)
    tree = take_action(tree, int_rules[15], 9)

    ans = tree[1][2]
end

# This problem is unsolved
function prob5()
    prob = [exp(x) * sin(x), x, 1, -1 // 2 * exp(x) * cos(x) + 1 // 2 * exp(x) * sin(x)]

end

function prob6()
    prob = [sqrt(9 - x^2) / x^2, x, 2, -asin(1 // 3 * x) - sqrt(9 - x^2) / x]
    que = Symbolics.value(∫(prob[1], ∂(x)))

    subsi = Vector{Dict{Num,Num}}()
    que = Symbolics.value(substitute(que, Dict(x => 3sin(x))))
    push!(subsi, Dict(x => 3sin(x)))
    tree = traverse_expr(que, returnTree=true)

    tree = take_action(tree, apply_∂, 17; args=(Symbolics.value(x),))
    take_action(tree, merge_args∫, 0)
    merge_args∫(tree[1][2])

    # See function unexpected to see what is problem in doing this
end

function prob7()
    prob = [1 / (x^2 * sqrt(4 + x^2)), x, 1, -1 // 4 * sqrt(4 + x^2) / x]
    que = Symbolics.value(∫(prob[1], ∂(x)))

    que = Symbolics.value(substitute(que, Dict(x => 2tan(x))))  # in this problem see 2tan(x) is no where in the node, but still action model has to figure this thing out and give it to the substituion model 
    subsi = Vector{Dict{Num,Num}}()
    push!(subsi, Dict(x => 2tan(x)))

    tree = traverse_expr(que, returnTree=true)

    tree = take_action(tree, take_common, 10, args=(4,))
    tree = take_action(tree, trigs_rules[6], 12)

    tree = take_action(tree, take_sqrt, 5)

    # this rule is quite common while doing substitution, (substitution, apply_∂, merge_args∫)  => we in future will look for model to combine tihs rule and create a new one for itself 
    tree = take_action(tree, apply_∂, 11, args=(Symbolics.value(x),))
    tree = take_action(tree, merge_args∫, 0)

    tree = take_action(tree, trigs_rules[6], 4)
    tree = take_action(tree, trigs_rules[2], 4)
    tree = take_action(tree, trigs_rules[1], 8)

    tree = take_action(tree, simplify, 0)   # a model will never exactly know what simplify will be doing 

    push!(subsi, Dict(sin(x) => t))

    # After substituting work on this 
    # I think it's better to calculate derivative of what we are substituting, and then checking how we can convert the whole expression to something else
    # apply_∂(Symbolics.value(∂(sin(x))), Symbolics.value(x)) / ∂(x)
    Symbolics.value(∫(1 / (t * (1 - t^2)), ∂(t)))
    substitute(tree[0+1][2], Dict(sin(x) => t, cos(x) => 1, ∂(x) => ∂(t)))
    tree = take_action(tree, substitute, 0; args=(Dict(sin(x) => t, cos(x) => 1, ∂(x) => ∂(t)),))

    tree = take_action(tree, const_out, 0)
    tree = take_action(tree, heur_int_rules[1], 2)

    # Reverse back the substitutions
    d = pop!(subsi)
    tree = take_action(tree, substitute, 0; args=(first(values(d)) => first(keys(d)),))

    # Make this substitution work from model
    tree = take_action(tree, substitute, 0; args=(Dict(sin(x) => (x / 2) / sqrt(1 + x^2 / 4)),))
    ans = tree[1][2]
end

function prob8()
    prob = [x / sqrt(3 - 2 * x - x^2), x, 3, asin(1 // 2 * (-1 - x)) - sqrt(3 - 2 * x - x^2)]
    que = Symbolics.value(∫(prob[1], ∂(x)))
    tree = traverse_expr(que, returnTree=true)

    tree = take_action(tree, complete_square, 4)

    subsi = Vector{Dict{Any,Any}}()
    push!(subsi, Dict((1 + x) => 2sin(t)))

    # Make sure to put Symbolics.value() inside the substituion dict
    tree = take_action(tree, substitute, 0; args=(Dict((x => Symbolics.value(2sin(t) - 1)),)))
    tree = take_action(tree, apply_∂, 17; args=(Symbolics.value(t),))
    tree = take_action(tree, merge_args∫, 0)

    # The substituion model will modify the trigs_rules[5] to apply here => we are skipping it for now
    take_common(tree[13+1][2], 4)
    xmx = 4 * cos(t)^2

    tree = take_action(tree, substitute, 0; args=(Dict(tree[13+1][2] => Symbolics.value(xmx))))
    tree = take_action(tree, take_sqrt, 12)
    tree = take_action(tree, const_out, 0)

    tree = take_action(tree, expand_integral_sum, 0)
    tree = take_action(tree, const_out, 8)
    tree = take_action(tree, const_out, 1)

    tree = take_action(tree, int_rules[5], 3)

    # Now reverse the substituions 
    tree[1][2]   # -t - 2.0cos(t)
    s = pop!(subsi) #  1 + x => 2sin(t)
    tree = take_action(tree, substitute, 0; args=(Dict(t => asin((1 + x) / 2), cos(t) => 1 // 2 * sqrt(3 - 2x - x^2)),))
    ans = tree[1][2]
end

# Finish this problem 
function prob9()
    prob = [sqrt(1 + x^2), x, 2, 1 // 2 * asinh(x) + 1 // 2 * x * sqrt(1 + x^2)]
    que = Symbolics.value(∫(prob[1], ∂(x)))

    tree = traverse_expr(que, returnTree=true)
    tree = take_action(tree, substitute, 0; args=(Dict(x => tan(t),)))
    tree = take_action(tree, apply_∂, 8; args=(Symbolics.value(t),))
    tree = take_action(tree, merge_args∫, 0)

    tree = take_action(tree, trigs_rules[6], 3)
    tree = take_action(tree, take_sqrt, 2)

    # The idea is to write sec(t)^3 = sec(t)^2 * sec(t) and then apply by parts, go ahead with this later 
    substitute(que, Dict(x => tan(t)))

end

function prob10()
    prob = [5 * x * sqrt(1 + x^2), x, 2, 5 // 3 * (1 + x^2)^(3 // 2)]
    que = Symbolics.value(∫(prob[1], ∂(x)))
    tree = traverse_expr(que, returnTree=true)

    subsi = Vector{Any}()
    push!(subsi, Dict(x => tan(t)))
    tree = take_action(tree, substitute, 0; args=(Dict(x => tan(t)),))
    tree = take_action(tree, apply_∂, 12; args=(Symbolics.value(t),))
    tree = take_action(tree, trigs_rules[6], 4)
    tree = take_action(tree, merge_args∫, 0)
    tree = take_action(tree, take_sqrt, 3)

    # Make the substituion work sec(t)=>x  and sec(t)*tan(t)dt => dx
    push!(subsi, Dict(sec(t) => x))
    substitute(tree[1][2], Dict(tan(t) * sec(t) => x))

    # After substituion
    tree = traverse_expr(Symbolics.value(5∫(x^2, ∂(x))), returnTree=true)
    tree = take_action(tree, int_rules[1], 2)

    # Reverse the substituion
    xmx = pop!(subsi)
    tree = take_action(tree, substitute, 0; args=(Dict(x => sec(t)),))

    xmx = pop!(subsi)
    xmx = Dict(sec(t) => sqrt(1 + x^2))
    tree = take_action(tree, substitute, 0; args=(xmx,))
    ans = tree[1][2]

end

function prob11()
    prob = [sqrt(-9 + exp(2 * t)), t, 4, -3 * atan(1 // 3 * sqrt(-9 + exp(2 * t))) + sqrt(-9 + exp(2 * t))]
    que = Symbolics.value(∫(prob[1], ∂(t)))
    tree = traverse_expr(que, returnTree=true)

    subsi = Vector{Vector{SymbolicUtils.BasicSymbolic{Real}}}()
    wrt = [[Symbolics.value(t), Symbolics.value(x)]]
    push!(subsi, [Symbolics.value(exp(t)), Symbolics.value(3sec(x))])

    der = apply_∂(subsi, wrt)

    # One of the Strategy for doing substituion is to first substitute in derivative then on the whole expression
    # Except ∂(t), transfer all the terms to RHS 
    der_sub = der[1][2] / (der[1][1] / occursin_operator(∂, der[1][1]))

    tree = take_action(tree, substitute, 0; args=(Dict(∂(t) => der_sub),))
    tree = take_action(tree, merge_args∫, 0)

    # Now substitute in the whole expression
    tree = take_action(tree, pow_rules[6], 9)
    take_action(tree, substitute, 0; args=(Dict(subsi[1][1] => subsi[1][2]),))

    tree[1][2]
    substitute(tree[1][2], Dict(subsi[1][1] => subsi[1][2]))  # handle this substituion using the model, either make exp(t)^2 or just square up 3sec(x) to get exp(2t)=>9sec(x)^2
    # the above code works in repl but not in the editor
    que = Symbolics.value(∫(tan(x) * sqrt(-9 + (9 // 1) * (sec(x)^2)), ∂(x)))
    tree = traverse_expr(que, returnTree=true)
    tree = take_action(tree, take_common, 5; args=(9,))

    # make the substituion model do something to make the below rule from trigs_rules[6]
    xmx = @birule -1.0 + sec(x)^2 ~ tan(x)^2
    tree = take_action(tree, xmx, 5)  # 
    tree = take_action(tree, take_sqrt, 4)
    tree = take_action(tree, const_out, 0)
    tree = take_action(tree, xmx, 3)
    tree = take_action(tree, expand_integral_sum, 2)
    tree = take_action(tree, int_rules[7], 7)

    tree = take_action(tree, const_out, 3)
    ans = tree[1][2]

    # Again make the reverse substituion work here
    xmx = pop!(subsi)
    xmx = Dict(tan(x) => sqrt(1 / 9 * exp(2t) - 1), x => atan(sqrt(1 / 9 * exp(2t) - 1)))
    tree = take_action(tree, substitute, 0; args=(xmx,))
    ans = tree[1][2]
end

# partial fraction problem
function prob12()
    # Partial fraction has to be solved by substituion model 
    # Partial fraction involves solving more than one linear equation having more than 1 variable
    prob = [(5 + x) // (-2 + x + x^2), x, 3, 2 * log(1 - x) - log(2 + x)]
    que = prob[1]


    # one partial fraction is done, then it's very simple to do the integration 
    
end

# yup
