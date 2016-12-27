using Documenter, CoupledFields

makedocs(
    modules = [CoupledFields],
    format = :html,
    sitename = "CoupledFields.jl",
    pages = Any[ 
        "Home" => "index.md",
        "Guide" => Any["man/Example1.md","man/Example2.md",],
        "Library" => "lib/library.md"
    ]
    )

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/Mattriks/CoupledFields.jl.git",
    julia  = "release",
    target = "build"
#    deps = nothing,
#    make = nothing
)
