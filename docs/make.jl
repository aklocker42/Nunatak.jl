using Documenter, Nunatak

makedocs(sitename = "Nunatak.jl",
         modules = [Nunatak],
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
         pages = ["Home" => "index.md",
                  "Validation" => "validation.md",
                  "API" => "api.md"],
         doctest = false,
         checkdocs = :exports)

deploydocs(repo = "github.com/aklocker42/Nunatak.jl.git", push_preview = true)
