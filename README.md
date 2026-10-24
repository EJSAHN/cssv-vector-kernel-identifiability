# cssv-vector-kernel-identifiability

Reproducible analysis code for a reanalysis of published cacao swollen shoot virus protective-ring trial counts. The workflow tests which protection and vector-kernel features are identifiable from aggregate zone-level field-trial data.

The repository produces tabular outputs only: CSV files and one Excel workbook. It does not contain figure-generation code.

## What is included

```text
data/source_tables/        transcribed source tables and reconstructed design counts
data/simulation/           year effects used to generate the reported simulation study
src/cssv_kernel_identifiability/
                           Python modules for data processing, likelihood fitting and simulation
scripts/run_analysis.py    command-line entry point
results/                   generated outputs, ignored by Git
```

`data/processed/` is not committed. It is generated when the workflow runs.

## Data provenance and reuse

The empirical count data are manual transcriptions of the published T1-T3 aggregate tables from Agusto et al. (2024), which analyses the CRIG field trial reported by Domfeh et al. (2019). The repository includes only small transcribed numeric source tables and reconstructed design counts needed to reproduce the analysis; it does not redistribute article PDFs, article figures, or table images.

Please cite the original field-trial and modelling sources when using these data:

- Domfeh O, Ameyaw GA, Dzahini-Obiatey HK, del Río Mendoza LE. 2019. Spatiotemporal spread of Cacao swollen shoot virus severe strain 1A in mixed hybrid cacao pre-inoculated with mild strain N1. *Plant Disease* 103, 3244-3250. doi:10.1094/PDIS-12-18-2175-RE.
- Agusto FB, Leite MCA, Owusu-Ansah F, Domfeh O, Hritonenko N, Chen-Charpentier B. 2024. Cacao sustainability: the case of cacao swollen-shoot virus co-infection. *PLoS ONE* 19, e0294579. doi:10.1371/journal.pone.0294579.

## Installation

```bash
conda create -n cssv_kernel python=3.10 -y
conda activate cssv_kernel
pip install -r requirements.txt
```

An existing Python environment with the packages in `requirements.txt` can also be used.

## Run the empirical analysis only

Use the reported denominators:

```bash
python scripts/run_analysis.py --skip-simulation --denominator-mode reported --output-dir results/reported_denominators
```

Use the reconstructed plot denominators:

```bash
python scripts/run_analysis.py --skip-simulation --denominator-mode reconstructed --output-dir results/reconstructed_denominators
```

## Run the empirical analysis plus simulation study

```bash
python scripts/run_analysis.py --denominator-mode reported --n-sim 200 --year-effects data/simulation/year_effects_calibration.csv --output-dir results/full_analysis
```

The simulated epidemics are generated from year effects calibrated on the empirical year-only beta-binomial fit. `data/simulation/year_effects_calibration.csv` contains the calibrated values used for the reported simulation study; with this file and the default seed (20260626), the simulated epidemics are regenerated exactly. Without `--year-effects`, the year effects are recalibrated in the run; small numerical differences in that fit between software environments change individual simulated outcomes, so replicate-level results then differ from the reported ones by Monte Carlo variation. The zone-level beta-binomial profiles are nearly flat across candidate scales, so the selected scale can also differ between environments for a small number of replicates.

The main output is:

```text
results/full_analysis/cssv_analysis_tables.xlsx
```

Each workbook sheet is also written as a CSV file under:

```text
results/full_analysis/csv/
```

In the simulation outputs, `profile_completed` is true when every candidate kernel scale has finite information criteria and the finite coefficient estimates needed for the summaries; only such replicates enter the recovery summaries, and `n_candidate_scales_fitted` gives the number meeting these requirements. `selected_optimizer_success` records the convergence flag returned by the optimiser for the selected candidate. An empty scale-selection table is returned when no replicate has a complete profile.

## Optimiser check

```bash
python scripts/run_analysis.py --denominator-mode reported --n-sim 200 --year-effects data/simulation/year_effects_calibration.csv --check-optimizer --output-dir results/full_analysis_checked
```

With `--check-optimizer`, the empirical model ladder and every candidate model in every simulated replicate are also refitted with tighter optimisation. Binomial refits use Fisher scoring with a target maximum absolute score below 1e-9 and report whether that tolerance was reached; they can also stop at the iteration or step-size limit. Beta-binomial models are refitted from three starting values by L-BFGS-B followed by a bounded trust-region search, both with analytic gradients, and the best solution is kept.

For each beta-binomial refit, the maximum absolute gradient and a curvature-scaled gradient diagnostic named `newton_decrement` are computed. The latter is half the sum of squared gradient projections divided by the absolute eigenvalues of a numerically differentiated Hessian; directions with absolute eigenvalue at most 1e-10 are omitted. Both diagnostics exclude log(kappa) when it is close to either bound. They do not check the one-sided optimality condition for that excluded coordinate, and a small value does not certify convergence or bound the remaining likelihood improvement. Optimiser status flags and the stability of the fitted results should be considered alongside these diagnostics; false flags are not confined to boundary solutions.

A replicate is compared only when every candidate has finite information criteria and coefficient estimates in both fits and no candidate raised an error. Excluded candidates are counted in `n_candidates_failed`. Empirical outputs retain fit messages; simulation outputs contain replicate-level counts and diagnostic maxima, not individual candidate messages or parameters. The option substantially increases the run time of the simulation study and adds the following sheets:

- `optimizer_check_empirical_<mode>`: original and refitted log-likelihoods, coefficient changes and refit diagnostics for each empirical model.
- `optimizer_check_replicates`: for each simulated replicate, the selected scale, support set and pressure-coefficient sign from the original fits and from the refits, with refit diagnostics.
- `optimizer_check_summary`: the same comparison summarised by generating scale and analysis unit.

These sheets compare the original fitting routine, run in the current session, with the refits. To compare the refits with previously reported selections (for example, the SimulationReplicates worksheet of the published supplementary data), merge `optimizer_check_replicates` with the reported table on the generating scale, the simulation identifier and the analysis unit, and compare `selected_scale_m_refit` with the reported selected scale. Because the original fitting routine can return slightly different optima in different software environments when candidate scales are nearly tied, the two comparisons can differ for a small number of zone-level replicates.

## Interpretation

The source-pressure covariate is a geometry-derived exposure score from outer CSSV-1A inoculum borders to monitored zones. It is not an observed mealybug dispersal distribution and should not be interpreted as a species-specific vector movement estimate.

## License

Code in this repository is released under the MIT License. The transcribed source tables are derived from cited open-access publications and are provided here for reproducibility with attribution to the original sources.
