using Documenter, DocumenterCitations, DocumenterVitepress, Literate, Nunatak

repository = joinpath(@__DIR__, "..")
generated = joinpath(@__DIR__, "src", "literated")
figures = joinpath(@__DIR__, "src", "assets", "figures")
bibliography = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :authoryear)

# The scripts run for minutes to days, so the pages show their code without executing it, plus committed figures.
examples = ["ice_cap" => "Growing an ice cap",
            "marine_ice_sheet" => "A marine ice sheet advancing into the ocean",
            "friction_inversion" => "Inverting for basal friction",
            "triangle_ice_shelf" => "An ice shelf in a bay on a triangle mesh"]

benchmarks = ["halfar" => "Halfar dome",
              "eismint2_a" => "EISMINT II experiment A",
              "halfpipe_streamice" => "STREAMICE halfpipe",
              "halfpipe_streamice_adjoint" => "STREAMICE halfpipe adjoint",
              "mismip_plus" => "MISMIP+",
              "kleiner2015_a" => "Polythermal slab, surface temperature cycle",
              "kleiner2015_b" => "Polythermal slab, steady state",
              "square_shelf_triangle" => "Square ice shelf on triangles against DJUICE"]

# A figure belongs to the longest script name it starts with, so halfpipe_streamice skips halfpipe_streamice_adjoint's.
function figure_owner(file)
    names = filter(n -> file == n * ".png" || startswith(file, n * "_"), first.(vcat(examples, benchmarks)))
    return isempty(names) ? nothing : argmax(length, names)
end

function figure_markdown(name)
    files = filter(f -> endswith(f, ".png") && figure_owner(f) == name, isdir(figures) ? readdir(figures) : String[])
    return join(("\n![$(f)](../assets/figures/$(f))\n" for f in files))
end

function add_heading(script, title, source)
    heading, body = startswith(script, "# # ") ? split(script, '\n'; limit = 2) : ("# # $(title)", script)
    return heading * "\n#\n# Source: `$(source)`\n#\n" * body
end

function literate_pages(directory, scripts)
    return map(scripts) do (name, title)
        Literate.markdown(joinpath(repository, directory, name * ".jl"), generated;
                          flavor = Literate.CommonMarkFlavor(), execute = false, credit = false,
                          preprocess = s -> add_heading(s, title, "$(directory)/$(name).jl"),
                          postprocess = s -> s * figure_markdown(name))
        return title => joinpath("literated", name * ".md")
    end
end

example_pages = literate_pages("examples", examples)
benchmark_pages = literate_pages("benchmarks", benchmarks)

format = DocumenterVitepress.MarkdownVitepress(repo = "github.com/aklocker42/Nunatak.jl",
                                               devbranch = "main",
                                               devurl = "dev")

makedocs(sitename = "Nunatak.jl",
         authors = "Nunatak.jl contributors",
         modules = [Nunatak],
         format = format,
         pages = ["Home" => "index.md",
                  "Model description" => ["Overview" => "model/overview.md",
                                          "Stress balances" => "model/stress_balances.md",
                                          "Rheology and sliding" => "model/rheology_and_sliding.md",
                                          "Grounding lines and calving fronts" => "model/grounding_and_fronts.md",
                                          "Thermodynamics" => "model/thermodynamics.md",
                                          "Triangle meshes" => "model/triangle_mesh.md",
                                          "Numerics" => "model/numerics.md"],
                  "Gradients and adjoints" => "gradients.md",
                  "Examples" => example_pages,
                  "Validation" => ["Overview" => "validation.md", benchmark_pages...],
                  "API" => "api.md",
                  "References" => "references.md"],
         plugins = [bibliography],
         doctest = false,
         checkdocs = :exports)

DocumenterVitepress.deploydocs(repo = "github.com/aklocker42/Nunatak.jl.git",
                               target = "build",
                               branch = "gh-pages",
                               devbranch = "main",
                               push_preview = true)
