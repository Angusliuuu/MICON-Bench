# MICON-Bench

MICON-Bench is a multi-image benchmark for evaluating image generation quality with a unified, checkpoint-based judge pipeline.

## Repository Structure

- `Evaluation/`: evaluation pipeline (prompt building, judging, aggregation)
- `DAR/`: auxiliary materials and assets

## Requirements

- Python 3.9+
- Install dependencies:
  - `pip install openai pyyaml pillow`

## Data (Hugging Face)

The dataset is **not** included in this repository. Download it from Hugging Face and place it under:

```
MICON-Bench-Release/
  Evaluation/
    data/
      object_composition/
      spatial_geometric_constraints/
      local_element_transfer/
      attribute_style_decoupling/
      foreground_background_composition/
      story_infer/
```

Replace the placeholder below with your Hugging Face dataset URL:

```
https://huggingface.co/<your-dataset-repo>
```

## Configure Paths

Edit `Evaluation/configs/global.yaml` to set paths (relative or absolute):

- `data_root`
- `generations_root`
- `eval_results_root`

## Configure Judge Models

Edit `Evaluation/configs/judge_models.yaml`. The default judge is **Qwen3-VL-32B-Instruct**.

### Default (local OpenAI-compatible endpoint)

```
judges:
  - name: "local-qwen-judge"
    provider: "http"
    model: "/path/to/Qwen3-VL-32B-Instruct"
    endpoint: "http://localhost:8000/v1"
    api_key_env: ""
```

### Add another MLLM

1. Add a new entry under `judges`:

```
  - name: "my-judge"
    provider: "http"
    model: "your-model-name-or-path"
    endpoint: "http://localhost:8000/v1"
    api_key_env: "MY_API_KEY"
```

2. Set it as default or per-task:

```
routing:
  default_judge: "my-judge"
  per_task:
    story_infer: "my-judge"
```

Notes:
- The endpoint must be **OpenAI-compatible** (chat/completions API).
- If `api_key_env` is set, export it before running:
  - `export MY_API_KEY=...`

## Expected Inputs

Place generated images under:

```
Evaluation/outputs/generations/<model>/<task>/<case_id>.png
```

Supported extensions: `.png`, `.jpg`, `.jpeg`.

Supported tasks (6 total):
- `object_composition`
- `spatial_geometric_constraints`
- `local_element_transfer`
- `attribute_style_decoupling`
- `foreground_background_composition`
- `story_infer`

## Run Evaluation

From `Evaluation/`:

```
bash run_evaluation.sh <MODEL_NAME> <TASK_TYPE|all> <JUDGE_NAME>
```

Examples:

```
bash run_evaluation.sh GPT-5-Image story_infer local-qwen-judge
bash run_evaluation.sh --all all local-qwen-judge
```

Outputs are saved to:

```
Evaluation/outputs/eval_res/<MODEL_NAME>/<MODEL_NAME>_<TIMESTAMP>/
```

## Optional: Candidate Evaluation (story_infer)

If you use candidate-based scoring for `story_infer`, place the template at:

```
Evaluation/data/story_infer/simple_candidate_template.json
```

To disable candidate evaluation when aggregating scores:

```
python scripts/aggregate_scores.py --model-name <MODEL_NAME> --no-candidate-eval
```

