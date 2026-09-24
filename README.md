# Bottlenecks of the Fly Brain

Max-flow / min-cut analysis of the male *Drosophila* CNS connectome, validated with in-silico ablation in a spiking (LIF) simulation.

**Question:** which neurons are structural bottlenecks between a sensory modality and the descending neurons that drive movement, and does silencing them hurt the motor response more than silencing the same number of random or high-centrality neurons?

## Status

- [x] 0. Setup
- [ ] 1. Ingestion
- [ ] 2. Graph and EDA
- [ ] 3. Source and sink sets
- [ ] 4. Max-flow / min-cut
- [ ] 5. Simulation and ablation
- [ ] 6. Visualization
- [ ] 7. Streamlit app
- [ ] 8. Write-up

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then paste your neuPrint token
```

Get the token from https://neuprint.janelia.org/account (log in with Google).

## Layout

```
data/raw/        cached neuPrint pulls (parquet), gitignored
data/processed/  cleaned graph, source/sink sets, results
src/             pipeline modules
app/             Streamlit app (reads data/processed only)
notebooks/       EDA only
reports/         figures and write-up
```

## Data

Connectome from neuPrint (Janelia / Google Research), male CNS. Details added once ingestion is done.
