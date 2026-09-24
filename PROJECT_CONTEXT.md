# PROJECT_CONTEXT: Bottlenecks of the Fly Brain

Max-flow / min-cut analysis of the male fruit fly connectome (Google Research + HHMI Janelia, male CNS v1.0, released Sept 2026), validated with in-silico ablation in a spiking simulation.

## Research question

Which neurons are structural bottlenecks between a given sensory modality (vision, taste, touch, etc.) and the descending neurons that drive movement? And does silencing those neurons in simulation hurt the motor response more than silencing the same number of neurons chosen randomly or by classic centrality?

The project is framed as: hypothesis (graph structure), prediction (min-cut set), test (ablation in simulation). That framing is the point of the portfolio piece.

## Data

- Source: neuPrint (neuprint.janelia.org), queried with `neuprint-python`. Login with Google, API token from the account page.
- Token lives in `.env` as `NEUPRINT_APPLICATION_CREDENTIALS`, never committed.
- Do not hardcode the dataset name. List available datasets from the client first and pick the male CNS one explicitly.
- Do not assume what columns mean. Inspect the neuron and connection tables first, report which annotations exist (cell type, class, predicted neurotransmitter, side, region) and how complete they are.
- Optional comparison dataset: FlyWire (female brain), used by the Shiu et al. 2024 LIF model.

## Hardware and environment

- Home: Windows, Ryzen 7 2700X, 16GB RAM, RTX 3080 (10GB).
- Laptop: ThinkPad L15, Ryzen 5 PRO 7530U, 16GB.
- 16GB RAM is the main constraint. The full edge list may not fit comfortably in memory in Python objects. Store as parquet, load with explicit dtypes (int32/int64 ids, int16/int32 weights), and support a synapse-count threshold everywhere.
- Do not invent library versions. Check what is installed or ask.

## Architecture

```
fly-bottlenecks/
  data/raw/           cached neuPrint pulls (parquet), gitignored
  data/processed/     cleaned graph, source/sink sets, results
  src/ingest.py       neuPrint client, cached queries
  src/graph.py        graph building, thresholds, subgraphs
  src/sets.py         sensory and descending neuron sets
  src/flow.py         max-flow / min-cut, centrality baselines
  src/sim.py          LIF simulation and ablation experiments
  src/viz.py          3D rendering, figures, Neuroglancer links
  app/streamlit_app.py
  notebooks/          EDA only, not part of the pipeline
  reports/            figures and write-up
```

Everything downstream reads from `data/processed`. The Streamlit app never calls neuPrint live, it only reads precomputed results.

## Modules (build one at a time, test before moving on)

### 0. Setup
git init, venv, `.gitignore` (data/, .env, caches), `requirements.txt` filled from what actually gets installed, README skeleton.

### 1. Ingestion
- Connect to neuPrint, list datasets, pick male CNS.
- Pull neuron table (all annotation columns that exist) and neuron-to-neuron connection table with synapse counts.
- Cache every query result to parquet. Re-running must not hit the API if the cache exists (a `--refresh` flag forces it).
- Pull in batches, the API is rate limited.
- Test: row counts, number of unique neurons vs published ~166,700, total synapses vs published ~125M (counts may differ depending on thresholds and what is included, report the difference instead of hiding it).

### 2. Graph and EDA
- Build a directed weighted graph (weight = synapse count).
- Report: degree distributions (median, quartiles, tail), weight distribution, share of edges below 3/5/10 synapses, coverage of neurotransmitter predictions.
- Build excitatory-only subgraph (based on predicted neurotransmitter, document the mapping used and the uncertainty).
- Test: memory footprint at each threshold, graph loads in under a reasonable time.

### 3. Source and sink sets
- Sources: sensory neurons per modality, from annotations.
- Sinks: descending neurons.
- Test: counts per modality, manual spot check of a few neurons in Neuroglancer.

### 4. Max-flow / min-cut
- Super-source connected to all sensory neurons of a modality, super-sink from all descending neurons, infinite capacity on those edges.
- Two variants, compare both:
  - Edge cut with capacity = synapse count, then map cut edges to neurons.
  - Node cut via node splitting (each neuron split into in/out with an internal capacity edge). Try capacity = 1 (minimum number of neurons) and capacity = total excitatory output.
- networkx will be too slow at this scale. Benchmark igraph and OR-Tools (SimpleMaxFlow) on a thresholded graph first, then scale up.
- Baselines: top-k by betweenness (approximate, sampled), PageRank, out-strength, random.
- Robustness: does the cut set stay stable across synapse thresholds (3/5/10)? Report overlap (Jaccard).
- Test: on a small synthetic graph with a known min-cut, flow value matches.

### 5. Simulation and ablation
- LIF model based on Shiu et al. 2024 (repo `philshiu/Drosophila_brain_model`, Brian2). First reproduce it on its original data if feasible, then port to male CNS edge list.
- Sign weights by predicted neurotransmitter.
- Protocol: stimulate sensory set of a modality, measure descending neuron firing. Then ablate: min-cut set vs random (many seeds) vs top betweenness vs top out-strength, same k.
- Statistics: effect size and distribution over random seeds, not a single run.
- If RAM is not enough: threshold weak edges, or restrict to neurons within N hops of the source/sink sets, or try a GPU backend (Brian2CUDA / GeNN). Document which option was used.
- Test: unablated run is stable across seeds, known pathway (e.g. sugar taste to feeding-related output) behaves plausibly.

### 6. Visualization
Goal: results look like a finished piece, not notebook output.
- 3D neurons with `navis` (skeletons, not full meshes, meshes are too heavy), rendered in plotly. Show source set, sink set, and cut set in distinct colors, everything else as a faint context layer.
- Neuroglancer links: generate shareable state URLs that highlight the cut neurons, so anyone can open them in the browser.
- Sankey or layered flow diagram: sensory modality to intermediate layers to descending neurons, with the bottleneck highlighted.
- Ablation chart: motor response per ablation strategy, random baseline as a distribution (violin/box), min-cut as a marker.
- Activity animation: firing rates from the simulation over time, mapped onto the 3D neurons, exported as mp4/gif for the README and write-up.
- Consistent, restrained palette and typography. No default matplotlib look.
- Test: every figure regenerates from `data/processed` with one command.

### 7. Streamlit app
- Pick modality, see 3D bottleneck neurons, see ablation results, toggle neurons off and see precomputed effect.
- Loads only precomputed data, fast on first open.

### 8. Write-up
- Short blog-style post plus a PDF version. Structure: question, data, method, result, limitations.
- Limitations to state explicitly: static connectome (no activity data), neurotransmitter predictions are predictions, synapse count is a proxy for weight, LIF model is a simplification, the viral "fly plays Doom" demos are readout-trained reservoirs, not the same thing.

## Working rules
- Module by module. After each module: short note on how to test it standalone, and a commit suggestion.
- Cache anything that touches the neuPrint API.
- Comments in code only where they actually matter.
- Report numbers with the base they are measured against. If a result is within noise of the random baseline, say so.
