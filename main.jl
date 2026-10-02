#################################################################################################
######### What do you want to run ? 
#################################################################################################
const YEARS = 365.241
const PRECISION = 1
program = Dict(
    
    ### Output folder
    :out_folder => get(ENV, "NETSURV_OUTPUT", joinpath(@__DIR__, "..", "assets")),

    ### First graphs on copulas and in appendix. 
    :plot_copulas => true,

    ### Estimator's bootstrap
    :run_bootstrap => true,
    :analyse_bootstrap => true,
    :n => 500, # Size of the samples
    :N_boot => 1000,

    ## Burn in study 
    :run_burnin => true, 
    :analyse_burnin => true,

    ### Test's bootstrap 
    :run_test_bootstrap => true,
    :analyse_test_bootstrap => true,
    :N_boot_test => 1000,

    ### Example on ccolon/survexp_fr
    :run_example => true,
    :analyse_example => true,
    :run_boot_example_variance => true,
    :analyse_boot_example_variance => true,
    :analyse_test_on_example => true,
    :N_boot_example_variance => 200,
)
# Number only the enabled stages, in execution order.
run_stages = filter(stage -> program[first(stage)], [
    :plot_copulas => "Figures des copules",
    :run_bootstrap => "Simulations des estimateurs",
    :analyse_bootstrap => "Analyse des estimateurs et production des figures/tableaux",
    :run_burnin => "Étude de convergence selon la taille d'échantillon",
    :analyse_burnin => "Analyse de la convergence et production des figures",
    :run_test_bootstrap => "Simulations du test (scénarios 1 et 2)",
    :analyse_test_bootstrap => "Analyse des tests et production des figures/tableaux",
    :run_example => "Estimation sur les données réelles",
    :analyse_example => "Figures des estimations sur les données réelles",
    :run_boot_example_variance => "Bootstrap de la variance sur les données réelles",
    :analyse_boot_example_variance => "Analyse des variances et production des figures",
    :analyse_test_on_example => "Tests sur les données réelles et production des tableaux",
])
function log_stage(key)
    index = findfirst(stage -> first(stage) == key, run_stages)
    @info "Étape $index/$(length(run_stages)): $(last(run_stages[index]))"
end
for directory in ("cops", "simus", "simus/test1", "simus/test2", "example")
    mkpath(joinpath(program[:out_folder], directory))
end
#################################################################################################
######### Library
#################################################################################################
using NetSurvival, 
      RateTables, 
      Distributions, 
      Copulas, 
      DataFrames, 
      Plots, 
      Random, 
      Plots.PlotMeasures, 
      Serialization, 
      .Threads, 
      StatsPlots, 
      Measurements, 
      Latexify, 
      PrettyTables, 
      ForwardDiff, 
      Roots, 
      Dates, 
      LinearAlgebra, 
      ProgressMeter
#################################################################################################
######### Sampler
#################################################################################################
function sampler(E, 𝒞, age_min, age_max)
    # Population mortality distribution: 
    age = (age_min + rand() * (age_max-age_min))YEARS
    year = (1990 + rand() * 20)YEARS
    sex = rand((:male,:female))
    P = Life(slopop[sex], age, year)

    # Vector (E,P) distributions, including dependence structure: 
    EP = SklarDist(SurvivalCopula(𝒞,(1,2)), (E, P))

    # A bit of random censoring + administrative censoring at 15 years: 
    c = min(rand(Exponential(20YEARS)), 15YEARS)

    # Final observed times :
    e,p = rand(EP, 1)
    o = min(e,p)
    t = min(o,c)
    δ = c > o # 0 means censored
    γ = p > e # 0 means censored
    return (age=age, year=year, sex=sex, e=e, p=p, c=c, o=o, t=t, δ=δ, γ=γ)
end
sampler(n, E, 𝒞, age_max, age_min) = DataFrame((sampler(E,𝒞, age_max, age_min) for i in 1:n))
sampler(n,E,𝒞) = sampler(n,E,𝒞,35,75)
#################################################################################################
######### First plots to describe the copulas themselves
#################################################################################################
clay(τ) = ClaytonCopula(2,Copulas.τ⁻¹(ClaytonCopula, τ))
frank(τ) = FrankCopula(2,Copulas.τ⁻¹(FrankCopula, τ))
gumb(τ) = GumbelCopula(2, Copulas.τ⁻¹(GumbelCopula, τ))
const cops = [frank(-0.3), clay(-0.3), IndependentCopula(2), frank(0.3), clay(0.3)]
const unicode_cop_names = ["Frank(-0.3)", "Clayton(-0.3)", "Indep.", "Frank(0.3)", "Clayton(0.3)"]
function plot_a_cop(U,title)
    return scatter(U[1,:],U[2,:],
        label=:none, ms=1, axes=:none, dpi=600, bottom_margin=0mm,  left_margin = 0mm, 
        right_margin = 0mm,  top_margin = 0mm, axis=nothing, ticks=nothing, xlabel=title, guidefontsize=12
    )
end
if program[:plot_copulas]
    log_stage(:plot_copulas)
    p = plot(plot_a_cop.(rand.(cops,5000),unicode_cop_names)...,layout=grid(1,5),size=[1380,304], bottommargin=10Plots.mm )
    savefig(p,"$(program[:out_folder])/cops/showoff_cops.pdf")
    savefig(p,"$(program[:out_folder])/cops/showoff_cops.png")

    # A second plot. 
    other_cops = [clay(0.9), frank(0.9), gumb(0.9)]
    other_titles = ["(a) : Clayton Copula", "(b) : Frank Copula", "(c) : Gumbel Copula"]
    p = plot(plot_a_cop.(rand.(other_cops,5000),other_titles)...,layout=grid(1,3),size=[1350,440], bottommargin=10Plots.mm )
    savefig(p,"$(program[:out_folder])/cops/example_archimedeans.pdf")
    savefig(p,"$(program[:out_folder])/cops/example_archimedeans.png")
end
#################################################################################################
######### Bootstrap the estimator and extract results
#################################################################################################
function sample_and_estimate(C₀, C₁, n, ℒₑ, rt)
    try
        return fit(GenPoharPerme(C₁), @formula(Surv(t,δ)~1), sampler(n, ℒₑ, C₀), rt)
    catch e
        (e isa DomainError || e isa PosDefException ||
         (e isa ErrorException && startswith(e.msg, "Implicit update"))) || rethrow()

        return nothing # retain numerical failures; never silently redraw a dataset
    end
end
function ci(r,t)
    S = r(t)
    σ = sqrt(variance(r,t))
    χ = sqrt(quantile(Chisq(1),0.95))
    return exp(log(S) - σ * χ), exp(log(S) + σ * χ)
end
contains(ci,val) = ci[1] <= val <= ci[2]
@latexrecipe function f(x::Measurement)
    return :($(x.val) ± $(x.err))
end
function PrettyTables._latex_alignment(s::Symbol)
    if (s == :l) || (s == :L)
        return "l"
    elseif (s == :c) || (s == :C)
        return "c"
    elseif (s == :r) || (s == :R)
        return "r"
    elseif (s == :X)
        return "X"
    elseif (s == :Y) || (s == :Z)
        return "Y"
    else
        error("Invalid LaTeX alignment symbol: $s.")
    end
end
function save_table(tbl, filename; rounding = ft_printf("%.1f"), alignment = :X)
    open(filename, "w") do  f
        pretty_table(
            f,
            tbl;
            backend=Val(:latex),
            formatters=(rounding, (v,i,j) -> alignment[j]==:Y ? "\$$v\$" : LatexCell(latexify(v))),
            header = latexify.(names(tbl)),
            alignment=alignment,
            tf = LatexTableFormat(
                top_line       = "\\toprule",
                header_line    = "\\midrule",
                mid_line       = "\\midrule",
                bottom_line    = "\\bottomrule",
                left_vline     = "",
                mid_vline      = "",
                right_vline    = "",
                header_envs    = [],
                subheader_envs = ["texttt"],
               )
        )
    end
    s = read(filename, String)
    write(filename, replace(replace(s, 
            "\\begin{tabular}" => "\\begin{tabularx}{\\linewidth}",
            "\\end{tabular}" => "\\end{tabularx}",
            "C_0" => "\\mathcal C_0",
            "C_1" => "\\mathcal C",
            "left( ; " => "left( ",
            "\$Fminuscop\$" =>"\$\\texttt{Frank}(-0.3)\$",
            "\$Cminuscop\$" =>  "\$\\texttt{Clayton}(-0.3)\$",
            "\$Picop\$" => "Indep.",
            "\$Fpluscop\$" => "\$\\texttt{Frank}(0.3)\$",
            "\$Cpluscop\$" => "\$\\texttt{Clayton}(0.3)\$",
            "H₀" => "\$H_0\$",
            "H₁" => "\$H_1\$",
            "H₂" => "\$H_2\$",
            "b₅" => "\$\\mathrm{biais}(5)\$",
            "r₅" => "\$\\mathrm{rmse}(5)\$",
            "p₅" => "\$\\mathrm{ecr}(5)\$",
            "b₁₀" => "\$\\mathrm{biais}(10)\$",
            "r₁₀" => "\$\\mathrm{rmse}(10)\$",
            "p₁₀" => "\$\\mathrm{ecr}(10)\$",
            "b₁₅" => "\$\\mathrm{biais}(15)\$",
            "r₁₅" => "\$\\mathrm{rmse}(15)\$",
            "p₁₅" => "\$\\mathrm{ecr}(15)\$",
            "\\\$" => "\$",
        ), 
        "\$\$" => "",
        "Fminuscop" =>"\$\\texttt{Frank}(-0.3)\$",
        "Cminuscop" =>  "\$\\texttt{Clayton}(-0.3)\$",
        "Picop" => "Indep.",
        "Fpluscop" => "\$\\texttt{Frank}(0.3)\$",
        "Cpluscop" => "\$\\texttt{Clayton}(0.3)\$",
    ))
end
const TRUE_E = Exponential(10YEARS)
const EVAL_TIMES = 0:5:(15YEARS)
if program[:run_bootstrap]
    log_stage(:run_bootstrap)
    Random.seed!(123)
    Sₑ5, Sₑ10, Sₑ15 = ccdf.(TRUE_E, [5YEARS, 10YEARS, 15YEARS])
    space = collect(Iterators.product(eachindex(cops),eachindex(cops),1:program[:N_boot]))
    models = Vector{Any}(undef,length(space))
    @showprogress Threads.@threads for ii in eachindex(space)
        (j,i,k) = space[ii]
        r = sample_and_estimate(cops[i], cops[j], program[:n], TRUE_E, slopop)
        models[ii] = (
            C₀ = i,
            C₁ = j,
            failed = isnothing(r),
            S = isnothing(r) ? fill(NaN, length(EVAL_TIMES)) : r.(EVAL_TIMES),
            biais5  = isnothing(r) ? NaN : r(5YEARS) .- Sₑ5,
            biais10 = isnothing(r) ? NaN : r(10YEARS) .- Sₑ10,
            biais15 = isnothing(r) ? NaN : r(15YEARS) .- Sₑ15,
            is_in5  = isnothing(r) ? NaN : contains(ci(r, 5YEARS), Sₑ5),
            is_in10 = isnothing(r) ? NaN : contains(ci(r, 10YEARS), Sₑ10),
            is_in15 = isnothing(r) ? NaN : contains(ci(r, 15YEARS), Sₑ15),
        )
    end
    models = DataFrame(models)
    @info "Estimator failures" failures=count(models.failed) total=nrow(models)
    serialize("$(program[:out_folder])/simus/my_serial_models.ser",models)
end
if program[:analyse_bootstrap]
    log_stage(:analyse_bootstrap)
    models = deserialize("$(program[:out_folder])/simus/my_serial_models.ser")

    rez_b = combine(groupby(models,["C₀","C₁"]), 
        :biais5 => mean => :b₅,
        :biais10 => mean => :b₁₀,
        :biais15 => mean => :b₁₅,
    )
    rez_r = combine(groupby(models,["C₀","C₁"]), 
        :biais5  =>  (x -> sqrt(mean(x .^ 2))) => :r₅,
        :biais10 => (x -> sqrt(mean(x .^ 2))) => :r₁₀,
        :biais15 => (x -> sqrt(mean(x .^ 2))) => :r₁₅,
    )
    rez_p = combine(groupby(models,["C₀","C₁"]), 
        :is_in5 => mean => :p₅,
        :is_in10 => mean => :p₁₀,
        :is_in15 => mean => :p₁₅,
    )

    working_names = ["Fminuscop","Cminuscop","Picop","Fpluscop","Cpluscop"]

    rez_b.C₀ = getindex.(Ref(working_names),rez_b.C₀)
    rez_b.C₁ = getindex.(Ref(working_names),rez_b.C₁)
    rez_r.C₀ = getindex.(Ref(working_names),rez_r.C₀)
    rez_r.C₁ = getindex.(Ref(working_names),rez_r.C₁)
    rez_p.C₀ = getindex.(Ref(working_names),rez_p.C₀)
    rez_p.C₁ = getindex.(Ref(working_names),rez_p.C₁)

    nms = names(unstack(rez_b, :C₀, :C₁, :b₅))
    rez5 = DataFrame(nms .=> ("b₅", "", "", "", "", ""))
    append!(rez5, unstack(rez_b, :C₀, :C₁, :b₅), promote=true)
    append!(rez5, DataFrame(nms .=> ("&&&&&\\\\[-5pt] r₅", "", "", "", "", "")))
    append!(rez5, unstack(rez_r, :C₀, :C₁, :r₅), promote=true)
    append!(rez5, DataFrame(nms .=> ("&&&&&\\\\[-5pt] p₅", "", "", "", "", "")))
    append!(rez5, unstack(rez_p, :C₀, :C₁, :p₅), promote=true)
    

    rez10 = DataFrame(nms .=> ("b₁₀", "", "", "", "", ""))
    append!(rez10, unstack(rez_b, :C₀, :C₁, :b₁₀), promote=true)
    append!(rez10, DataFrame(nms .=> ("&&&&&\\\\[-5pt] r₁₀", "", "", "", "", "")))
    append!(rez10, unstack(rez_r, :C₀, :C₁, :r₁₀), promote=true)
    append!(rez10, DataFrame(nms .=> ("&&&&&\\\\[-5pt] p₁₀", "", "", "", "", "")))
    append!(rez10, unstack(rez_p, :C₀, :C₁, :p₁₀), promote=true)

    rez15 = DataFrame(nms .=> ("b₁₅", "", "", "", "", ""))
    append!(rez15, unstack(rez_b, :C₀, :C₁, :b₁₅), promote=true)
    append!(rez15, DataFrame(nms .=> ("&&&&&\\\\[-5pt] r₁₅", "", "", "", "", "")))
    append!(rez15, unstack(rez_r, :C₀, :C₁, :r₁₅), promote=true)
    append!(rez15, DataFrame(nms .=> ("&&&&&\\\\[-5pt] p₁₅", "", "", "", "", "")))
    append!(rez15, unstack(rez_p, :C₀, :C₁, :p₁₅), promote=true)

    rez5.C₀ = Latexify.LaTeXString.(rez5.C₀)
    rez10.C₀ = Latexify.LaTeXString.(rez10.C₀)
    rez15.C₀ = Latexify.LaTeXString.(rez15.C₀)

    nms_to_change = ["", working_names...]

    rename!(rez5, nms_to_change)
    rename!(rez10, nms_to_change)
    rename!(rez15, nms_to_change)

    save_table(rez5,"$(program[:out_folder])/simus/rez5.tex"; rounding=ft_printf("%.4f"), alignment = [:l,:Y,:Y,:Y,:Y,:Y])
    save_table(rez10,"$(program[:out_folder])/simus/rez10.tex", rounding=ft_printf("%.4f"), alignment = [:l,:Y,:Y,:Y,:Y,:Y])
    save_table(rez15,"$(program[:out_folder])/simus/rez15.tex", rounding=ft_printf("%.4f"), alignment = [:l,:Y,:Y,:Y,:Y,:Y])

    # Now the graph : 
    function mkplot(sdf; recenter=false)
        i₀       = sdf.C₀[1]
        i₁       = sdf.C₁[1]
        is_left    = i₁ == 1
        is_right   = i₁ == length(cops)
        is_bottom  = i₀ == length(cops)
        is_top     = i₀ == 1
        true_curve = ccdf.(TRUE_E,EVAL_TIMES)

        if recenter
            p = errorline(
                EVAL_TIMES/YEARS, 
                hcat(sdf.S...) ./ true_curve, 
                errorstyle              =:plume, 
                left_margin             = 5mm,
                right_margin            = is_right ?  2mm : -2mm,
                bottom_margin           = is_bottom ?  5mm : -2mm,
                top_margin              = is_top ?  2mm : -2mm,
                ylabel                  = is_left ? "C₀: $(unicode_cop_names[i₀])" : "",
                xlabel                  = is_bottom ? "C: $(unicode_cop_names[i₁])" : "",
                guidefontsize           = 12,
            )
        else
            p = errorline(
                EVAL_TIMES/YEARS, 
                hcat(sdf.S...) ./ (recenter ? true_curve : 1), 
                errorstyle              =:plume, 
                ygrid                   = true, 
                yticks                  = true, 
                ytickfontsize           = is_left ? 6 : 1,
                y_foreground_color_text = is_left ? :black : :transparent,
                left_margin             = is_left ?  5mm : -2mm,
                right_margin            = is_right ?  2mm : -2mm,
                xtickfontsize           = is_bottom ? 6 : 1,
                x_foreground_color_text = is_bottom ? :black : :transparent,
                bottom_margin           = is_bottom ?  5mm : -2mm,
                top_margin              = is_top ?  2mm : -2mm,
                ylabel                  = is_left ? "C₀: $(unicode_cop_names[i₀])" : "",
                xlabel                  = is_bottom ? "C: $(unicode_cop_names[i₁])" : "",
                guidefontsize           = 12,
            )
            plot!(p, EVAL_TIMES/YEARS, ccdf.(TRUE_E, EVAL_TIMES))
        end
        return p
    end
    rrr = [mkplot(sdf, recenter=true) for sdf in groupby(models,[:C₀,:C₁])]
    p2 = plot(rrr..., legend=false, size=(1200,800));
    savefig(p2,"$(program[:out_folder])/simus/featherplot.pdf")
    savefig(p2,"$(program[:out_folder])/simus/featherplot.png")

    rrr = [mkplot(sdf) for sdf in groupby(models,[:C₀,:C₁])]
    p = plot(rrr..., legend=false, size=(1200,800), ylims=(0,1));
    savefig(p,"$(program[:out_folder])/simus/featherplot2.pdf")
    savefig(p,"$(program[:out_folder])/simus/featherplot2.png")

    
end
if program[:run_burnin]
    log_stage(:run_burnin)
    Random.seed!(123)
    # Sₑ5, Sₑ10, Sₑ15 = ccdf.(TRUE_E, [5YEARS, 10YEARS, 15YEARS])
    Λ5, Λ10, Λ15 = log.(ccdf.(TRUE_E, [5YEARS, 10YEARS, 15YEARS]))
    space = collect(Iterators.product(eachindex(cops),eachindex(cops), [250, 500, 1000, 2000, 5000]))
    models = Vector{Any}(undef,length(space))
    @showprogress Threads.@threads for ii in eachindex(space)
        (j,i,k) = space[ii]
        r = sample_and_estimate(cops[i], cops[j], k, TRUE_E, slopop)
        models[ii] = (
            C₀ = i,
            C₁ = j,
            n = k,
            failed = isnothing(r),
            b5  = isnothing(r) ? NaN : log(r(5YEARS)) - Λ5,
            b10 = isnothing(r) ? NaN : log(r(10YEARS)) - Λ10,
            b15 = isnothing(r) ? NaN : log(r(15YEARS)) - Λ15,
            r5 = isnothing(r) ? NaN : variance(r,5YEARS),
            r10 = isnothing(r) ? NaN : variance(r,10YEARS),
            r15 = isnothing(r) ? NaN : variance(r,15YEARS),
        )
    end
    models = DataFrame(models)
    @info "Burn-in failures" failures=count(models.failed) total=nrow(models)
    serialize("$(program[:out_folder])/simus/my_serial_burnin.ser",models)
end
if program[:analyse_burnin]
    log_stage(:analyse_burnin)
    models = deserialize("$(program[:out_folder])/simus/my_serial_burnin.ser")
    function mk_burnin_plot(sdf; which_one=:biais, M = 1)
        i₀       = sdf.C₀[1]
        i₁       = sdf.C₁[1]
        is_left    = i₁ == 1
        is_right   = i₁ == length(cops)
        is_bottom  = i₀ == length(cops)
        is_top     = i₀ == 1

        if which_one == :biais
            p = plot(sdf.n, abs.(sdf.b5), 
                left_margin             = 5mm,
                right_margin            = is_right ?  2mm : -2mm,
                bottom_margin           = is_bottom ?  5mm : -2mm,
                top_margin              = is_top ?  2mm : -2mm,
                ylabel                  = is_left ? "C₀: $(unicode_cop_names[i₀])" : "",
                xlabel                  = is_bottom ? "C: $(unicode_cop_names[i₁])" : "",
                guidefontsize           = 12,
                label="t = 5",
                ylims=(0,M),
                legend = is_top && is_right ? true : false,
            )
            p = plot!(sdf.n, abs.(sdf.b10), label="t = 10)")
            p = plot!(sdf.n, abs.(sdf.b15), label="t = 15)")
        elseif which_one == :rmse
            p = plot(sdf.n, sqrt.(sdf.r5 .+ sdf.b5.^2), 
                left_margin             = 5mm,
                right_margin            = is_right ?  2mm : -2mm,
                bottom_margin           = is_bottom ?  5mm : -2mm,
                top_margin              = is_top ?  2mm : -2mm,
                ylabel                  = is_left ? "C₀: $(unicode_cop_names[i₀])" : "",
                xlabel                  = is_bottom ? "C: $(unicode_cop_names[i₁])" : "",
                guidefontsize           = 12,
                label="t = 5",
                ylims=(0,M),
                legend = is_top && is_right ? true : false,
            )
            p = plot!(sdf.n, sqrt.(sdf.r10 .+ sdf.b10.^2), label="t = 10")
            p = plot!(sdf.n, sqrt.(sdf.r15 .+ sdf.b15.^2), label="t = 15")
        end
        return p
    end
    finite_max(values) = maximum(filter(isfinite, values); init=0.01)
    M = finite_max(vcat(abs.(models.b5), abs.(models.b10), abs.(models.b15)))
    rrr = [mk_burnin_plot(sdf, which_one=:biais, M=M) for sdf in groupby(models,[:C₀,:C₁])]
    p2 = plot(rrr..., size=(1200,800));
    savefig(p2,"$(program[:out_folder])/simus/burnin_biais.pdf")
    savefig(p2,"$(program[:out_folder])/simus/burnin_biais.png")

    M = finite_max(vcat(sqrt.(models.r5 .+ models.b5.^2), sqrt.(models.r10 .+ models.b10.^2), sqrt.(models.r15 .+ models.b15.^2)))
    rrr = [mk_burnin_plot(sdf, which_one=:rmse, M=M) for sdf in groupby(models,[:C₀,:C₁])]
    p2 = plot(rrr..., legend=false, size=(1200,800));
    savefig(p2,"$(program[:out_folder])/simus/burnin_rmse.pdf")
    savefig(p2,"$(program[:out_folder])/simus/burnin_rmse.png")
end
#################################################################################################
######### Bootstrap the test and extract results
#################################################################################################
function sample_and_test(C₀₁, C₀₂, C₁, n, ℒₑ₁, ℒₑ₂, rt; different_ps = false)
    try
        half_n = Int(n/2)
        if different_ps
            df = vcat(sampler(half_n, ℒₑ₁, C₀₁, 35, 65), sampler(half_n, ℒₑ₂, C₀₂, 65, 75))
        else
            df = vcat(sampler(half_n, ℒₑ₁, C₀₁), sampler(half_n, ℒₑ₂, C₀₂))
        end
        df.group = [ones(Int64,half_n)...,2*ones(Int64,half_n)...]
        return fit(GraffeoTest(C₁),@formula(Surv(t,δ)~group), df, rt)
    catch e
        (e isa DomainError || e isa PosDefException ||
         (e isa ErrorException && startswith(e.msg, "Implicit update"))) || rethrow()

        return nothing # failed draws must not disappear from calibration summaries
    end
end
if program[:run_test_bootstrap]
    log_stage(:run_test_bootstrap)
    Random.seed!(123)
    eachindex(cops)
    E1 = Exponential(5YEARS)
    Hs = ["H₀","H₁","H₂"]
    E2s = Exponential.([5YEARS,6YEARS,10YEARS])
    space = collect(Iterators.product(
        eachindex(cops),
        eachindex(cops),
        eachindex(Hs),
        1:2, # test version
        1:program[:N_boot_test],
    ))
    tests = Vector{Any}(undef,length(space))
    @showprogress Threads.@threads for ii in eachindex(space)
        (j,i,k,version,_) = space[ii]
        r = sample_and_test(cops[i], cops[i], cops[j], program[:n], E1, E2s[k], slopop, different_ps = (version==2))
        tests[ii] = (
            H = Hs[k],
            C₀ = i,
            C₁ = j,
            v = version,
            failed = isnothing(r),
            p5 = isnothing(r) ? NaN : pvalue(r, 5YEARS),
            p10 = isnothing(r) ? NaN : pvalue(r, 10YEARS),
            p15 = isnothing(r) ? NaN : pvalue(r, 15YEARS),
        )
    end
    tests = DataFrame(tests)
    @info "Test failures" failures=count(tests.failed) total=nrow(tests)
    serialize("$(program[:out_folder])/simus/my_serial_tests.ser",tests)
end
function mktble(tests, sym)
    return unstack(
        combine(
            groupby(
                tests,
                [:H, :C₀, :C₁]
            ), 
            sym => (x -> 100*(mean(x)± (1.96 * std(x) / sqrt(length(x))))) => :reject_counts
        ),
        :H,
        :reject_counts
    )
end
function mkhisto15(sdf)
        
    pvals = sdf.p15
    ok = .! isnan.(pvals)

    i₀         = sdf.C₀[1]
    i₁         = sdf.C₁[1]
    is_left    = i₁ == 1
    is_right   = i₁ == length(cops)
    is_bottom  = i₀ == length(cops)
    is_top     = i₀ == 1

    return plot(
        histogram(pvals[ok], bins=50,normalize=:pdf),
        # ylims=(0,0.08),
        ylabel = is_left ? "C₀: $(unicode_cop_names[i₀])" : "",
        xlabel = is_bottom ? "C: $(unicode_cop_names[i₁])" : "",
        guidefontsize  = 12,
        bottom_margin  = is_bottom ?  5mm : -2mm,
        top_margin = is_top ?  2mm : -2mm,
        left_margin  = is_left ?  5mm : -2mm,
        right_margin = is_right ?  2mm : -2mm,
    )
end
if program[:analyse_test_bootstrap]
    log_stage(:analyse_test_bootstrap)
    all_tests = deserialize("$(program[:out_folder])/simus/my_serial_tests.ser")
    two_boostraps = (
        (
            "$(program[:out_folder])/simus/test1",
            filter(:v => x -> x == 1, all_tests),
        ),
        (
            "$(program[:out_folder])/simus/test2",
            filter(:v => x -> x == 2, all_tests),
        ),
    )
    working_names = ["Fminuscop","Cminuscop","Picop","Fpluscop","Cpluscop"]
    nms_to_change = ["", working_names...]

    for (folder, tests) in two_boostraps
        p = plot(
            [mkhisto15(sdf) for sdf in groupby(filter(:H => x-> x == "H₀", tests), [:C₀,:C₁])]..., 
            legend=false, 
            size=(1200,800)
        )
        savefig(p,"$(folder)/histo_H0.pdf")
        savefig(p,"$(folder)/histo_H0.png")
        
        tests.C₁ = [working_names[i] for i in tests.C₁]
        tests.C₀ = [working_names[i] for i in tests.C₀]
        tests.reject5 = ifelse.(isfinite.(tests.p5), tests.p5 .< 0.05, NaN)
        tests.reject10 = ifelse.(isfinite.(tests.p10), tests.p10 .< 0.05, NaN)
        tests.reject15 = ifelse.(isfinite.(tests.p15), tests.p15 .< 0.05, NaN)

        tst5 = mktble(tests, :reject5)
        tst10 = mktble(tests, :reject10)
        tst15 = mktble(tests, :reject15)

        nms = names(unstack(tst5, :C₀, :C₁, :H₀))
        rez5 = DataFrame(nms .=> ("H₀", "", "", "", "", ""))
        append!(rez5, unstack(tst5, :C₀, :C₁, :H₀), promote=true)
        append!(rez5, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₁", "", "", "", "", "")))
        append!(rez5, unstack(tst5, :C₀, :C₁, :H₁), promote=true)
        append!(rez5, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₂", "", "", "", "", "")))
        append!(rez5, unstack(tst5, :C₀, :C₁, :H₂), promote=true)
        
        rez10 = DataFrame(nms .=> ("H₀", "", "", "", "", ""))
        append!(rez10, unstack(tst10, :C₀, :C₁, :H₀), promote=true)
        append!(rez10, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₁", "", "", "", "", "")))
        append!(rez10, unstack(tst10, :C₀, :C₁, :H₁), promote=true)
        append!(rez10, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₂", "", "", "", "", "")))
        append!(rez10, unstack(tst10, :C₀, :C₁, :H₂), promote=true)

        rez15 = DataFrame(nms .=> ("H₀", "", "", "", "", ""))
        append!(rez15, unstack(tst15, :C₀, :C₁, :H₀), promote=true)
        append!(rez15, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₁", "", "", "", "", "")))
        append!(rez15, unstack(tst15, :C₀, :C₁, :H₁), promote=true)
        append!(rez15, DataFrame(nms .=> ("&&&&&\\\\[-5pt] H₂", "", "", "", "", "")))
        append!(rez15, unstack(tst15, :C₀, :C₁, :H₂), promote=true)

        rez5.C₀ = Latexify.LaTeXString.(rez5.C₀)
        rez10.C₀ = Latexify.LaTeXString.(rez10.C₀)
        rez15.C₀ = Latexify.LaTeXString.(rez15.C₀)

        rename!(rez5, nms_to_change)
        rename!(rez10, nms_to_change)
        rename!(rez15, nms_to_change)

        save_table(rez5,"$(folder)/tst5.tex"; rounding=ft_round(1), alignment = [:l,:Z,:Z,:Z,:Z,:Z])
        save_table(rez10,"$(folder)/tst10.tex"; rounding=ft_round(1), alignment = [:l,:Z,:Z,:Z,:Z,:Z])
        save_table(rez15,"$(folder)/tst15.tex"; rounding=ft_round(1), alignment = [:l,:Z,:Z,:Z,:Z,:Z])
    end
end
#################################################################################################
######### Example
#################################################################################################

# Load and prepare dataset: 
df = deepcopy(ccolon) # From NetSurvival.jl directly. 
select!(df, Not(:stage))

df_right = select(filter(:side => ==(:right), df), Not(:side))
df_left = select(filter(:side => ==(:left), df), Not(:side))

τ_frank = -0.6:0.2:0.6
τ_clayton = -0.3:0.1:0.3
T_FINAL = 10

if program[:run_example]
    log_stage(:run_example)
    models = Dict()
    @showprogress Threads.@threads for (τ,(df,side)) in collect(Iterators.product(τ_frank,((df_left, :left),(df_right, :right))))
            models[(side,:frank,τ)] = fit(GenPoharPerme(frank(τ)), @formula(Surv(time,status)~1), df, survexp_fr)
    end
    @showprogress Threads.@threads for (τ,(df,side)) in collect(Iterators.product(τ_clayton,((df_left, :left),(df_right, :right))))
            models[(side,:clayton,τ)] = fit(GenPoharPerme(clay(τ)), @formula(Surv(time,status)~1), df, survexp_fr)
    end
    serialize("$(program[:out_folder])/example/models.ser", models)
end
if program[:analyse_example]
    log_stage(:analyse_example)
    models = deserialize("$(program[:out_folder])/example/models.ser")

    function mkshortplot(models, side, cop, τs)
        t = 0:0.1:T_FINAL
        m = models[(side,cop,τs[1])]
        plt = plot(t, m.(t .* 365.241), label="τ=$(τs[1])", legend = :inline, ylims=(0.2,1))
        for i in 2:length(τs)
            m = models[(side,cop,τs[i])]
            plot!(plt, t, m.(t .* 365.241), label="τ=$(τs[i])", legend = :inline)
        end
        return plt
    end
    function mkshortvarplot(models, side, cop, τs)
        t = 0:0.1:T_FINAL
        m = models[(side,cop,τs[1])]
        plt = plot(t, sqrt.(variance.(Ref(m), t .* 365.241)), label="τ=$(τs[1])", legend = :inline, ylims=(0,0.18))
        for i in 2:length(τs)
            m = models[(side,cop,τs[i])]
            plot!(plt, t, sqrt.(variance.(Ref(m), t .* 365.241)), label="τ=$(τs[i])", legend = :inline)
        end
        return plt
    end

    f_pl = plot(mkshortplot(models, :left,  :frank,   τ_frank), ylabel="C: Frank(τ)")
    f_pr = plot(mkshortplot(models, :right, :frank,   τ_frank))
    c_pl = plot(mkshortplot(models, :left,  :clayton, τ_clayton), ylabel="C: Clayton(τ)", xlabel="Primary tumor location: left")
    c_pr = plot(mkshortplot(models, :right, :clayton, τ_clayton), xlabel="Primary tumor location: right")
    allp = plot(f_pl, f_pr, c_pl, c_pr,layout=(2,2),size=(1200,650), leftmargin=5Plots.mm, bottommargin=5Plots.mm, guidefontsize = 12)
    Plots.savefig(allp,"$(program[:out_folder])/example/short_paper_graph.pdf")
    Plots.savefig(allp,"$(program[:out_folder])/example/short_paper_graph.png")

    f_pl = plot(mkshortvarplot(models, :left,  :frank,   τ_frank), ylabel="C: Frank(τ)")
    f_pr = plot(mkshortvarplot(models, :right, :frank,   τ_frank))
    c_pl = plot(mkshortvarplot(models, :left,  :clayton, τ_clayton), ylabel="C: Clayton(τ)", xlabel="Primary tumor location: left")
    c_pr = plot(mkshortvarplot(models, :right, :clayton, τ_clayton), xlabel="Primary tumor location: right")
    allp = plot(f_pl, f_pr, c_pl, c_pr,layout=(2,2),size=(1200,650), leftmargin=5Plots.mm, bottommargin=5Plots.mm, guidefontsize = 12)
    Plots.savefig(allp,"$(program[:out_folder])/example/short_paper_graph_var.pdf")
    Plots.savefig(allp,"$(program[:out_folder])/example/short_paper_graph_var.png")
end
N = program[:N_boot_example_variance]
eval_times = 1:1:10YEARS
cl_cops = clay.(τ_clayton)
J, K, L = length(eval_times), length(cl_cops), nrow(df_left)
if program[:run_boot_example_variance]
    log_stage(:run_boot_example_variance)
    # Bootstrap to get unbiaised estimators of the standard errors: 
    boot, b = zeros(N,K,J), rand(1:L,(N,L))
    ProgressMeter.@showprogress Threads.@threads for (i,k) in collect(Iterators.product(1:N,1:K))
        boot[i,k,:] = fit(GenPoharPerme(cl_cops[k]), @formula(Surv(time,status)~1), df_left[b[i,:],:], survexp_fr).(eval_times)
    end
    serialize("$(program[:out_folder])/example/boot.ser",boot)
end
if program[:analyse_boot_example_variance]
    log_stage(:analyse_boot_example_variance)
    Random.seed!(123)

    # Get estimated standard errors:  
    σ_est = zeros(K,J)
    ProgressMeter.@showprogress Threads.@threads for k in 1:K
        model = fit(GenPoharPerme(cl_cops[k]), @formula(Surv(time,status)~1), df_left, survexp_fr)
        σ_est[k,:] = sqrt.(variance.(Ref(model), eval_times))
    end

    # Get bootstrapped ones: 
    boot = deserialize("$(program[:out_folder])/example/boot.ser")
    σ_boot = sqrt.(dropdims(var(log.(boot),dims=1),dims=1))

    function mkplottitrage(σ_est, σ_boot; recentered=0)
        percentages = σ_est ./ σ_boot
        if recentered > 0
            percentages .-= percentages[recentered:recentered,:]             # the idea is that the independent estimator should be truth
        end
        plt = plot(legend=:bottom);
        for i in eachindex(τ_clayton)
            plot!(plt, eval_times ./ YEARS, percentages[i,:], label="τ = $(τ_clayton[i])")
        end
        return plt
    end

    p1 =  plot(mkplottitrage(σ_est, σ_boot), ylims=(0.5, 1.5)); #
    p2 = plot(mkplottitrage(σ_est, σ_boot; recentered = findfirst(τ_clayton.==0.0)),ylims=(-0.3,0.3)); #
    plt = plot(plot(p1,p2), size=(1200,450))
    Plots.savefig(plt,"$(program[:out_folder])/example/bootstrap_variance.pdf")
    Plots.savefig(plt,"$(program[:out_folder])/example/bootstrap_variance.png")
end
if program[:analyse_test_on_example]
    log_stage(:analyse_test_on_example)
    mk_example_test(df, tbl, cop) = fit(GraffeoTest(cop), @formula(Surv(time,status)~side), df, tbl)
    Ts = [2,3,5,8,10] .* YEARS
    times = ["T = $(Int(i/YEARS))" for i in Ts]

    # Claytons
    τ_clayton = -0.3:0.1:0.3
    clay_test = mk_example_test.(Ref(df),Ref(survexp_fr),clay.(τ_clayton))
    clay_pvals = DataFrame(pvalue.(clay_test, Ts'), times)
    insertcols!(clay_pvals, 1, :τ => τ_clayton)
    save_table(clay_pvals, "$(program[:out_folder])/example/tests_pvalues_clayton.tex"; rounding=ft_printf("%.5f"), alignment = [:r,:Y,:Y,:Y,:Y,:Y])

    # Franks 
    τ_frank = -0.6:0.1:0.6
    frank_test = mk_example_test.(Ref(df),Ref(survexp_fr),frank.(τ_frank))
    frank_pvals = DataFrame(pvalue.(frank_test, Ts'), times)
    insertcols!(frank_pvals, 1, :τ => τ_frank)
    save_table(frank_pvals, "$(program[:out_folder])/example/tests_pvalues_frank.tex"; rounding=ft_printf("%.5f"), alignment = [:r,:Y,:Y,:Y,:Y,:Y])
end

@info "Run terminé ($(length(run_stages)) étapes)."
