# Bottlenecks of the Fly Brain

Max-flow / min-cut analysis of the male *Drosophila* CNS connectome, validated with in-silico ablation in a spiking (LIF) simulation.

**Question:** which neurons are structural bottlenecks between a sensory modality and the descending neurons that drive movement, and does silencing them hurt the motor response more than silencing the same number of random or high-centrality neurons?

## Status

- [x] 0. Setup
- [x] 1. Ingestion
- [x] 2. Graph and EDA
- [x] 3. Source and sink sets
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

Connectome from neuPrint (Janelia FlyEM, Cambridge, Google Connectomics), dataset `male-cns:v1.0`
(last database edit 2026-03-28, segment property update 2026-06-08).

```powershell
python src/inspect_schema.py --dataset male-cns:v1.0   # read-only schema probe
python src/ingest.py --dataset male-cns:v1.0           # cached pull; --refresh to re-pull, --report for counts only
```

The full pull takes about 10 minutes. It writes `neurons.parquet` (11 MB) and `edges/` (163 MB) to `data/raw/male-cns_v1.0/`.

| | pulled | published |
|---|---|---|
| `:Neuron` nodes | 176,422 | |
| neurons with `superclass` | 166,700 | ~166,700 |
| neuron→neuron edges | 25,862,574 | |
| synapses (sum of `weight`) | 125,024,863 | ~125M |
| synapses (sum of `weightHP`) | 107,079,478 | |

Edges kept at each synapse threshold: ≥3: 10.6M (105.0M synapses), ≥5: 6.3M (90.3M), ≥10: 2.8M (67.5M).

## Graph

```powershell
python src/graph.py --dataset male-cns:v1.0   # builds data/processed/male-cns_v1.0/
python src/eda.py --dataset male-cns:v1.0     # writes eda.json there
```

174,386 neurons with at least one edge and 25,862,452 edges (122 self-loops dropped). `weightHR` equals `weight` on every edge, so it is not kept. The whole graph loads as a sparse matrix in about 1 s and takes 208 MB. At ≥5 synapses it takes 51 MB.

**Neurotransmitter sign** (`consensusNt`, by presynaptic neuron): acetylcholine = excitatory; GABA, glutamate and histamine = inhibitory; dopamine, serotonin, octopamine and unclear = no sign. That gives a known sign for 93.9% of neurons. 59% of synapses are cholinergic, and those form the excitatory subgraph. These are predictions, not measurements. Median prediction confidence is 0.96 for ACh, 0.86 for GABA and 0.81 for glutamate. `consensusNt` agrees with the per-neuron `predictedNt` for 88.4% of neurons. Glutamate is inhibitory at most central synapses but not all.

Caveat for the excitatory subgraph: all 6,093 photoreceptors (`ol_sensory`) are histaminergic, so an ACh-only graph has no edges leaving the visual sensory set.

## Source and sink sets

```powershell
python src/sets.py --dataset male-cns:v1.0   # writes sets.parquet and sets_report.json (with Neuroglancer spot-check links)
```

Sources come from the `class` / `subclass` / `superclass` annotations (definitions in `src/sets.py`). Sinks are the 1,314 neurons with `superclass == descending_neuron`; motor neurons (815) are kept as an alternative readout.

| modality | neurons | output synapses | share going directly to DNs | median hops to DNs |
|---|---|---|---|---|
| vision (photoreceptors) | 6,086 | 684,494 | 0.0% | 3 |
| olfaction (ORNs) | 2,639 | 1,364,154 | 0.04% | 2 |
| taste, head | 275 | 169,010 | 3.4% | 2 |
| taste, legs/wings | 1,153 | 540,333 | 2.1% | 2 |
| Johnston's organ | 512 | 231,113 | 15.9% | 2 |
| head mechanosensory (all) | 1,652 | 576,636 | 21.5% | 2 |
| touch, body | 2,558 | 1,395,875 | 1.6% | 2 |
| proprioception | 1,454 | 1,005,570 | 1.7% | 2 |
| hygrosensation | 66 | 55,757 | 1.0% | 2 |
| thermosensation | 25 | 50,522 | 0.9% | 2 |

From every modality, all 1,314 descending neurons can be reached.
