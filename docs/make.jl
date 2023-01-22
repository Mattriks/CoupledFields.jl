using Documenter, CoupledFields, Cairo

makedocs(
    modules = [CoupledFields],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    clean = false,
    sitename = "CoupledFields.jl",
    pages = [
        "Home" => "index.md",
        "Guide" => ["man/Example1.md", "man/Example2.md", "man/Example3.md"],
        "Library" => "lib/library.md"
    ],
)


deploydocs(
    repo = "github.com/Mattriks/CoupledFields.jl.git",
)
