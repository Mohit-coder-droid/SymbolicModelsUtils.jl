# using CairoMakie
# using GraphMakie
# using Graphs


"""For traversing symbolic expressions, by depth-first search"""
# function traverse_expr(expr, parent=-1, current=-1; returnTree::Bool=false, tree=nothing, printTree::Bool=false)
# current += 1
# if SymbolicUtils.iscall(expr)
#     arg, op = arguments(expr), operation(expr)

#     (printTree) && println("Operation: $op  Parent: $parent  Current: $current")

#     # Store the tree
#     if (returnTree)
#         push!(tree, (parent, current, op))
#     end

#     parent = current

#     for a in arg
#         current = traverse_expr(a, parent, current, returnTree=returnTree, tree=tree, printTree=printTree)
#     end

#     return current
# else
#     (printTree) && println("Operation: $expr  Parent: $parent  Current: $current")

#     if (returnTree)
#         push!(tree, (parent, current, expr))
#     end

#     return current
# end
# end

"""For visualizing of symbolic expression in terms of graphs"""
function visualizeTree(expr)
    tree = Vector{Tuple{Int,Int,Any}}()

    traverse_expr(ex, returnTree=true, tree=tree, printTree=false)  # printTree

    g = SimpleGraph(length(tree))

    for e in tree
        if e[1] >= 0
            add_edge!(g, e[1] + 1, e[2] + 1)
        end
    end

    node_labels = [string(e[3]) for e in tree]
    node_colors = [:blue for i in 1:nv(g)]
    node_colors[1] = :red

    f, ax, p = graphplot(g,
        nlabels=node_labels,
        node_color=node_colors)
    display(f)
end


# function justExperiment()

# visualizeTree(ex)


# for e in tree
#     if e[1] >= 0
#         add_edge!(g, e[1] + 1, e[2] + 1)
#     end
# end

# node_labels = [string(e[3]) for e in tree]
# node_colors = [:blue for i in 1:nv(g)]
# node_colors[1] = :red
# f, ax, p = graphplot(g,
#     nlabels=node_labels)



# # Add edges (bi-directional by default)

# # add_edge!(g, 2, 5)  # add a diagonal connection

# # Optional: define node and edge colors
# node_colors = [:red, :orange, :green, :blue, :purple]
# edge_colors = fill(:gray, ne(g))

# # Create the plot
# f, ax, p = graphplot(g,
#     node_color=node_colors,
#     nlabels=repr.(1:nv(g)),
#     edge_color=edge_colors,
#     node_labels=1:nv(g))  # label nodes with their IDs

# # offsets = 0.15 * (p[:node_pos][] .- p[:node_pos][][1])
# # p.nlabels_offset[] = offsets

# hidedecorations!(ax)
# hidespines!(ax)
# autolimits!(ax)
# ax.aspect = DataAspect()

# f  # Display the figure




# g = wheel_graph(10)
# f, ax, p = graphplot(g)
# hidedecorations!(ax)
# hidespines!(ax)
# ax.aspect = DataAspect()

# using GraphMakie.NetworkLayout
# g = SimpleGraph(5)
# add_edge!(g, 1, 2)
# add_edge!(g, 2, 4)
# add_edge!(g, 4, 3)
# add_edge!(g, 3, 2)
# add_edge!(g, 2, 5)
# add_edge!(g, 5, 4)
# add_edge!(g, 4, 1)
# add_edge!(g, 1, 5)

# # define some edge colors
# edgecolors = [:black for i in 1:ne(g)]
# edgecolors[4] = edgecolors[7] = :red

# f, ax, p = graphplot(g, layout=Shell(),
#     node_color=[:black, :red, :red, :red, :black],
#     edge_color=edgecolors)
# hidedecorations!(ax)
# hidespines!(ax)
# ax.aspect = DataAspect()
# f

# end