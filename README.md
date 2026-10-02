This repo contains code for the paper "Non-parametric estimation of net survival under dependence between death causes":

```bibtex
@misc{laverny2025nonparametricestimationnetsurvival,
      title={Non-parametric estimation of net survival under dependence between death causes}, 
      author={Oskar Laverny and Nathalie Grafféo and Roch Giorgi},
      year={2025},
      eprint={2502.09273},
      archivePrefix={arXiv},
      primaryClass={math.ST},
      url={https://arxiv.org/abs/2502.09273}, 
}
```

This code leverages the [`NetSurvival.jl`](https://github.com/JuliaSurv/NetSurvival.jl) package, in which the core algorithms are written.

The Manifest tracks the remote NetSurvival branch
`fix/generalized-inference-performance`; no local `Pkg.develop` is needed.
From the paper repository root, with Julia 1.10.0:

```powershell
julia +1.10.0 --startup-file=no --project=code
```

Then run `using NetSurvival` followed by
`include(joinpath(dirname(pathof(NetSurvival)), "..", "test", "generalized_inference.jl"))`
to check the package corrections, or rerun the experiments in a fresh process:

```powershell
julia +1.10.0 --startup-file=no --project=code -t auto code/main.jl
```

The full run uses 1000 simulation replicates and 200 real-data bootstrap resamples.
The mesh is fixed at one day. New results go to `assets/`; the previous results are archived in `assets_submitted/`.
Numerical failures are retained and reported. Old serialized estimators must
be regenerated to use the corrected variances.
