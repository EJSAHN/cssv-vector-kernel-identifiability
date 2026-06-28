# cssv-vector-kernel-identifiability

Reproducible analysis code for a reanalysis of published cacao swollen shoot virus protective-ring trial counts. The workflow tests which protection and vector-kernel features are identifiable from aggregate zone-level field-trial data.

The repository produces tabular outputs only: CSV files and one Excel workbook. It does not contain figure-generation code.

## What is included

```text
data/source_tables/        transcribed source tables and reconstructed design counts
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
python scripts/run_analysis.py --denominator-mode reported --n-sim 200 --output-dir results/full_analysis
```

The main output is:

```text
results/full_analysis/cssv_analysis_tables.xlsx
```

Each workbook sheet is also written as a CSV file under:

```text
results/full_analysis/csv/
```

## Interpretation

The source-pressure covariate is a geometry-derived exposure score from outer CSSV-1A inoculum borders to monitored zones. It is not an observed mealybug dispersal distribution and should not be interpreted as a species-specific vector movement estimate.

## License

Code in this repository is released under the MIT License. The transcribed source tables are derived from cited open-access publications and are provided here for reproducibility with attribution to the original sources.
