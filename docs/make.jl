using Documenter, SymbolicModelsUtils

DocMeta.setdocmeta!(SymbolicModelsUtils, :DocTestSetup, :(using SymbolicModelsUtils); recursive=true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers

function nice_name(file)
    file = replace(file, r"^[0-9]*-" => "")
    if haskey(page_rename, file)
        return page_rename[file]
    end
    return splitext(file)[1] |> x -> replace(x, "-" => " ") |> titlecase
end

makedocs(;
    modules=[SymbolicModelsUtils],
    doctest=true,
    linkcheck=false, # Rely on Lint.yml/lychee for the links
    authors="Mohit Sahu <mohitsahuandmohitsahu@gmail.com> and contributors",
    repo="https://github.com/mohit-coder-droid/SymbolicModelsUtils.jl/blob/{commit}{path}#{line}",
    sitename="SymbolicModelsUtils.jl",
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://mohit-coder-droid.github.io/SymbolicModelsUtils.jl"
    ),
    pages=[
        "Home" => "index.md",
        "Equation Model" => Any[
            "equation/model.md",
            "equation/dataset.md",
            "equation/examples.md"],
        "Integral Model" => Any["integral/model.md",
            "integral/dataset.md",
            "integral/examples.md",
            "integral/rules.md"],
        "API Reference" => "reference.md"
    ],
)

deploydocs(; repo="github.com/mohit-coder-droid/SymbolicModelsUtils.jl.git",
    devbranch="main",
    push_preview=false)