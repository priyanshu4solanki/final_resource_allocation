# Cloud Task Resource Allocation

An end-to-end research project for task resource prediction, dependency-aware
graph encoding, multi-agent scheduling, and folder-driven task execution.

## What's included

- A FastAPI hub, telemetry agent, and worker for coordinating Python tasks.
- A synthetic offline scheduling demo and trained MAPPO scheduler.
- Predictor, graph encoder, and scheduler checkpoints used by the live hub.
- Final MAPPO evaluation data, training history, and plots in
  `results/mappo_reference_1000/`.
- The cleaned cloud-task dataset and metadata, plus reproducible synthetic GNN
  training data.
- Unit and integration tests, and scripts for dataset preparation, training,
  and report-graph generation.

## Setup and tests

Requires Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s . -p "test*.py"
```

Run the synthetic offline demo with:

```powershell
python main_engine.py --tasks 8 --seed 7 --checkpoint-dir models
```

The MAPPO training script can reproduce a run and write its outputs to a
separate directory:

```powershell
python train_reference_mappo.py --episodes 1000 --tasks 8 --eval-seeds 20 --seed 42 --output-dir results/mappo_reference_reproduced
```

## Run the live hub

The hub loads the included predictor, GNN, and 1,000-episode MAPPO
checkpoints by default. Set an API token before starting it to enable task
submission:

```powershell
$env:MARL_API_TOKEN = "<long-random-secret>"
$env:MARL_NODE_IDS = "S1,S2,S3"
python hub.py
```

Open `http://localhost:8000/` for the dashboard. On each worker machine, point
the agent and task runner to the hub and to the shared task/result directories:

```powershell
$env:MARL_TELEMETRY_URL = "http://<hub-ip>:8000/api/telemetry"
$env:MARL_NODE_ID = "S1"
$env:MARL_TASKS_DIR = "<shared-task-directory>"
$env:MARL_RESULTS_DIR = "<shared-result-directory>"
python agent.py
python task_runner.py
```

Use matching unique node IDs and configure shared directories consistently on
the hub and workers. Run submitted scripts only on isolated, trusted workers;
the task runner executes them with the worker's OS permissions.

## Final evaluation

The included run trained for 1,000 episodes and evaluated on 20 held-out
seeds. Trained MAPPO achieved a mean reward of `0.4868`, with 100% completion
and no rejections. The fastest baseline scored `0.4884`, so MAPPO was slightly
lower in mean reward; these results do not show that it outperforms that
baseline. See `results/mappo_reference_1000/evaluation.json` for full metrics.

Evaluation is based on a local simulator, synthetic workloads, synthetic
deadlines, and a cost-derived energy proxy; it is not a production benchmark.
The supplied GNN training examples and graph links are synthetic. The cleaned
cloud dataset's GPU labels are heuristic pseudo-labels, not measured GPU
requirements. Review the accompanying metadata JSON files before using either
dataset for research or model training.
