using SMLMBoxer
using Documenter

DocMeta.setdocmeta!(SMLMBoxer, :DocTestSetup, :(using SMLMBoxer); recursive=true)

makedocs(;
    modules=[SMLMBoxer],
    authors="klidke@unm.edu",
    repo="https://github.com/JuliaSMLM/SMLMBoxer.jl/blob/{commit}{path}#{line}",
    sitename="SMLMBoxer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaSMLM.github.io/SMLMBoxer.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/SMLMBoxer.jl",
    devbranch="main",
)
