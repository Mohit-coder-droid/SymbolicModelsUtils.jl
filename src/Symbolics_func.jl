# Function that are picked up directly and modified slightly from Symbolics.jl 

function Base.numerator(x::SymbolicUtils.BasicSymbolic{Real})
    if iscall(x) && operation(x) == /
        x = arguments(x)[1] # get numerator
    end
    return x
end

function Base.denominator(x::SymbolicUtils.BasicSymbolic{Real})
    if iscall(x) && operation(x) == /
        x = arguments(x)[2] # get denominator
    else
        x = 1
    end
    return x
end
