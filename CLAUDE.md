# CLAUDE.md

Contributor/agent-facing guide to this repository. The `README.md` is the user-facing
documentation (step-by-step usage, IAM, dashboards) — this file complements it with
architecture, conventions, and the non-obvious facts you need before changing anything.

## Project overview

This is an AWS sample (`sample-ray-on-amazon-sagemaker-training-jobs`) showing how to run
**Ray** distributed workloads inside **Amazon SageMaker training jobs**. The centerpiece is
`scripts/launcher.py`: a universal entrypoint that bootstraps a Ray cluster on the SageMaker
instances allocated to a job and then runs the user's training/inference script with Ray
already initialized. Everything else in the repo (notebooks, example scripts, Grafana
dashboard) demonstrates how to use that launcher.

## Repository structure

```
scripts/
  launcher.py              # CANONICAL launcher — the single source of truth
  requirements.txt         # ray[data,train,tune,serve]==2.54.0, sagemaker==3.5.0
examples/<pattern>/<framework>/
  notebook.ipynb           # copies launcher.py in, then launches a ModelTrainer job
  scripts/                 # entry script + model + requirements.txt
grafana-dashboards/
  ray_sagemaker_training_dashboard.json
images/  README.md  CONTRIBUTING.md  LICENSE  CODE_OF_CONDUCT.md
```

Four Ray patterns, each with a homogeneous and a heterogeneous variant:

| Pattern            | Entry script   | What it shows                                    |
| ------------------ | -------------- | ------------------------------------------------ |
| `ray-remote`       | `train.py`     | `@ray.remote` task parallelism (sentiment clf)   |
| `ray-data`         | `inference.py` | Batch inference with `ray.data` (ResNet152)      |
| `ray-tune`         | `tune.py`      | HPO with ASHA on CIFAR-10                        |
| `ray-torchtrainer` | `train_ray.py` | LLM LoRA/QLoRA fine-tuning (HuggingFace + Train) |

Variant dirs are named `pytorch` / `pytorch-heterogeneous` (or `huggingface` /
`huggingface-heterogeneous` for `ray-torchtrainer`).

## The launcher (`scripts/launcher.py`)

**Single source of truth.** The README states users should not modify the launcher — it is a
universal entrypoint. Any change to launcher behavior belongs in `scripts/launcher.py`.

The same `launcher.py` runs on **every** instance in the job. It does not know in advance
whether it is the head or a worker — it discovers its role at runtime from the SageMaker
environment (`env.current_host` vs the computed `head_host`) and then follows the matching code
path. This is why a single file can drive single-node, homogeneous multi-node, and
heterogeneous clusters.

### File organization (top to bottom)

| Lines     | Section                | Contents                                                             |
| --------- | ---------------------- | -------------------------------------------------------------------- |
| 1–28      | Imports                | stdlib, `boto3`, `sagemaker_training`, `ray`, `requests`, `yaml`     |
| 31–59     | Logging                | `get_logger()` — single configured, non-propagating logger           |
| 61–81     | Constants & globals    | exit codes, ports, timeouts, `FAILURE_REASON_PATH`, mutable globals  |
| 84–112    | Signal handling        | `signal_handler()` + SIGTERM/SIGINT registration                     |
| 115–219   | EFA discovery          | `get_efa_supported_gpu_instances()` + fallback/RDMA instance lists   |
| 222–415   | Arg/env parsing        | `_parse_args()` — CLI args, env-var fallbacks, entrypoint splitting  |
| 418–648   | Prometheus helpers     | binary copy/extract, remote-write URL building & config injection    |
| 651–752   | Runtime env builder    | `_create_runtime_environment()` — env vars passed to Ray workers     |
| 755–915   | Entry-script execution | `_execute_entry_script()` + `.py` (importlib) / `.sh` (subprocess)   |
| 917–1252  | Process/util helpers   | host→IP, log readers, subprocess runners, command allowlist, untar   |
| 1255–1484 | Node setup             | `_setup_head_node()`, `_setup_worker_node()`                         |
| 1487–1571 | Cluster topology       | `_is_ray_alive()`, `_get_cluster_configuration()` + homo/hetero      |
| 1574–1845 | Orchestration          | single-node / multi-node setup + homo/hetero environment entrypoints |
| 1848–1930 | Main                   | `_write_failure_reason_file()`, `main()`, `__main__` guard           |

### Control flow

`main()` (L1859) → `_parse_args()` (L222) → `sagemaker_training.environment.Environment()` →
optional Prometheus binary extraction → branch on `env.is_hetero`:

- `_setup_ray_environment_homogeneous_cluster` (L1774) / `_setup_ray_environment_heterogeneous_cluster` (L1808)
- both call `_create_runtime_environment` (L651), then `_get_cluster_configuration` (L1502) → `(all_hosts, head_host, total_host_count)`
- 1 host → `_setup_single_node_ray` (L1574); else `_setup_multi_node_ray` (L1717)
- multi-node routes by `env.current_host == head_host` → `_setup_head_node` (L1255) or `_setup_worker_node` (L1450)

```
main
 ├─ _parse_args                      args + env-var fallbacks; sets source_dir/entry_script
 ├─ Environment()                    SageMaker cluster metadata (hosts, instance type, is_hetero)
 ├─ _copy_prometheus_binary          (only if --prometheus-path) extract offline binary
 └─ _setup_ray_environment_*_cluster
     ├─ _create_runtime_environment  build env-var dict for Ray (NCCL/EFA/RDMA/Prometheus)
     ├─ _get_cluster_configuration   → (all_hosts, head_host, total_host_count)
     ├─ total==1 → _setup_single_node_ray ─┐
     └─ else     → _setup_multi_node_ray   │
                    ├─ head → _setup_head_node ──┤ both: ray start → ray.init →
                    └─ else → _setup_worker_node │   [Prometheus] → _run_script → shutdown
```

### Phase-by-phase behavior

1. **Parse & normalize inputs** (`_parse_args`, L222). Reads CLI flags, then fills any unset
   value from the matching env var. Splits `-e/--entrypoint` (e.g. `training/train.py`) into
   `source_dir` + `entry_script` and exports them as env vars for later phases. Unknown args are
   logged and ignored (so user training args pass through harmlessly).
2. **Read SageMaker environment** (`main`, L1877). `Environment()` exposes `hosts`,
   `current_host`, `instance_groups_dict`, `is_hetero`, `num_cpus`, `num_gpus`,
   `network_interface_name`, `current_instance_type`.
3. **Build the Ray runtime env** (`_create_runtime_environment`, L651). Copies the full process
   environment, prepends `source_dir` to `PYTHONPATH`, sets NCCL vars, conditionally enables EFA
   /RDMA by instance type, and wires Prometheus/Grafana env vars. This dict is passed to
   `ray.init(runtime_env={"env_vars": ...})` so every Ray worker inherits it.
4. **Resolve topology** (`_get_cluster_configuration`, L1502). Homogeneous: head = `hosts[0]`,
   count = `len(hosts)`. Heterogeneous: head = first host of `--head-instance-group`; if that
   group is coordinator-only (`head_num_cpus==0 and head_num_gpus==0`) its hosts are excluded
   from the worker count.
5. **Start Ray for this node's role** (head/worker/single). See below.
6. **Run the user script** (`_run_script` → `_execute_entry_script`, L755) — only on the head /
   single node, after the cluster is assembled.
7. **Shut down** in a `finally` block: optional `--wait-shutdown` delay, `ray metrics
shutdown-prometheus`, `ray.shutdown()`, `ray stop`. On the worker, instead loop on `ray
status` until the head disappears, then `ray stop`.

### Head node, in detail (`_setup_head_node`, L1255)

1. Compute the head's resources: `num_cpus`/`num_gpus` default to `env.num_cpus`/`env.num_gpus`
   but are overridden by `--head-num-cpus`/`--head-num-gpus` when provided (L1275). These can be
   `0` for coordinator-only heterogeneous heads.
2. Build the start command:
   `ray start --head --num-cpus=<n> --num-gpus=<n> --port=6379`, appending
   `--dashboard-host=0.0.0.0 --dashboard-port=8265 --metrics-export-port=8080` when
   `--include-dashboard` is on (L1286). Values are `shlex.quote`d.
3. Run it via `_run_subprocess_command_with_env` (the runtime-env dict is merged into the Ray
   process environment), then `ray.init(address="auto", include_dashboard=..., runtime_env={"env_vars": runtime_env})` (L1303).
4. **Prometheus** (only if `include_dashboard and launch_prometheus`, L1305): if a remote host
   was detected, inject `remote_write` into the config _before_ launch; then start either the
   custom binary (`_build_prometheus_command`) or `ray metrics launch-prometheus`, async, with
   stdout/stderr redirected to `/tmp/prometheus_*.log`. Poll `GET <host>/-/healthy` every 2s up
   to `PROMETHEUS_WAIT_SECONDS=300`, bailing early if the process exits (which usually means no
   internet to download the binary). Failure here is logged, not fatal.
5. **Wait for workers** (L1408): loop reading `ray.available_resources()`, counting keys that
   start with `node:`, until `connected_nodes == cluster_size` or `RAY_CONNECTION_TIMEOUT=300s`
   elapses (after which it proceeds with whatever connected, logging a warning).
6. `_run_script(runtime_env)` executes the user entry script (L1431).
7. **`finally` teardown** (L1437): if no failure, honor `--wait-shutdown`; then
   `ray metrics shutdown-prometheus`, `_shutdown_ray_safely()` (`ray.shutdown()`), and
   `ray stop`. Returns the `ray stop` return code.

### Worker node, in detail (`_setup_worker_node`, L1450)

1. Resolve the head hostname to an IP with `_get_ip_from_host` (L917): `socket.gethostbyname`
   retried up to 200 times with a 5s sleep between attempts (~1000s max; SageMaker DNS can lag
   at startup).
2. `ray start --address=<ip>:6379` with the runtime env merged in (L1466).
3. Stay alive: loop `_is_ray_alive()` (which shells out to `ray status`, L1487) every
   `RAY_WORKER_POLL_INTERVAL=10s`, logging roughly once a minute, until the head disappears.
4. Then `ray stop` and return.

Workers never run the user script or Prometheus — they only join the cluster and keep the
process alive so SageMaker doesn't tear the instance down early.

### Single-node, in detail (`_setup_single_node_ray`, L1574)

Same shape as the head path but without the worker-wait loop: `ray start --head --port=6379`
(+ dashboard), `ray.init`, optional Prometheus, `_run_script`, then the same `finally` teardown.
Chosen when `total_host_count == 1`.

### Cluster topology resolution (`_get_cluster_configuration`, L1502)

- **Homogeneous** (`_get_homogeneous_cluster_config`, L1521): `head_host = hosts[0]`,
  `total_host_count = len(hosts)`.
- **Heterogeneous** (`_get_heterogeneous_cluster_config`, L1528): iterate
  `env.instance_groups_dict`, accumulate all hosts; the head is the first host of the group
  named by `--head-instance-group`. If that head is coordinator-only
  (`head_num_cpus==0 and head_num_gpus==0`), its hosts are subtracted from the worker count
  (L1555) so the head-wait math is correct. Raises if the named group isn't found.

### User entry script invocation (`_execute_entry_script`, L755)

Driven by env vars `source_dir` / `entry_script` (set from `-e/--entrypoint` in `_parse_args`).
Runs only on the head / single node, _after_ the cluster is assembled, so Ray is already
initialized when user code runs.

- Resolves the absolute script path under the SageMaker code dir (`/opt/ml/input/data/code`),
  logging directory contents for debugging, and raises `FileNotFoundError` if missing.
- `.py` → `_execute_python_script` (L841): `chdir` into the source dir, prepend it to
  `sys.path`, then load via `importlib.util.spec_from_file_location("__main__", path)` and
  `exec_module`. The module **must** guard real work behind `if __name__ == "__main__":`.
- `.sh` → `_execute_bash_script` (L868): `subprocess.run(["bash", path], env=runtime_env,
check=True, capture_output=True)`; stdout/stderr are logged.
- Any other extension → `ValueError`. On exception, sets `has_failure = True`, logs the
  traceback, and re-raises; a `finally` restores the original working directory.

### Runtime environment construction (`_create_runtime_environment`, L651)

Builds the env-var dict handed to `ray.init(runtime_env={"env_vars": ...})`, so it propagates to
every Ray worker process across the cluster:

- Starts from a full copy of the launcher's `os.environ`.
- Prepends the absolute `source_dir` to `PYTHONPATH` so user modules import on all nodes.
- Sets `NCCL_SOCKET_IFNAME = env.network_interface_name` and `NCCL_PROTO = simple`.
- If `current_instance_type` ∈ `SM_EFA_NCCL_INSTANCES` → `FI_PROVIDER=efa`.
- If ∈ `SM_EFA_RDMA_INSTANCES` (p4d/p4de/trn1) → `FI_EFA_USE_DEVICE_RDMA=1`, `RDMAV_FORK_SAFE=1`.
- Prometheus/Grafana wiring: when `launch_prometheus`, sets `RAY_PROMETHEUS_HOST` to
  `http://127.0.0.1:9090` for the Dashboard, and if the user-supplied `RAY_PROMETHEUS_HOST` is
  remote, stores it as `RAY_REMOTE_WRITE_PROMETHEUS_HOST` (plus passes through
  `RAY_PROMETHEUS_USERNAME`/`PASSWORD`). Also forwards `RAY_PROMETHEUS_NAME`, `RAY_GRAFANA_HOST`,
  and `RAY_GRAFANA_IFRAME_HOST` (defaulting the iframe host to `RAY_GRAFANA_HOST`).

The EFA-capable instance list is fetched live from EC2 `describe_instance_types` at _import
time_ (`get_efa_supported_gpu_instances`, L115), filtered to GPU instances, with a static
fallback (`SM_EFA_NCCL_INSTANCES_FALLBACK`, L176) if credentials/API fail.

### Prometheus / observability

Local Prometheus runs on `127.0.0.1:9090` and the Ray Dashboard points at it. Remote write is
controlled by `RAY_PROMETHEUS_HOST`. The flow on the head/single node:

1. `_extract_amp_region` (L538) regex-matches `aps-workspaces.<region>.amazonaws.com`.
2. `_build_remote_write_url` (L551) appends `/api/v1/remote_write` for AMP (region matched) or
   `/api/v1/write` for self-hosted, preserving any existing path on the host.
3. `_inject_remote_write_config` (L585) loads the prometheus.yml, appends a `remote_write` entry
   with `queue_config` (max_samples_per_send 1000, max_shards 200, capacity 2500), adds a
   `sigv4: {region}` block for AMP or a `basic_auth` block for self-hosted, and writes it back.
   Injection happens **before** Prometheus launches.

Config-path selection (`_get_prometheus_config_path`, L567): `ray metrics launch-prometheus`
uses the Ray package template path; a custom binary uses the session config
`/tmp/ray/session_latest/metrics/prometheus/prometheus.yml`. For offline clusters, pass a
pre-downloaded binary via `--prometheus-path` — `_copy_prometheus_binary` (L432) copies the
tar.gz into `/opt/ml/code`, verifies size, and extracts it with the traversal-safe
`_safe_extract_all` (L1196).

### Subprocess execution helpers (L992–1193)

All shelling-out is centralized and hardened:

- `_validate_command` (L992): allowlist — only `ray`, `bash`, `./prometheus-*`, or
  `/opt/ml/code/prometheus-*` may run; anything else raises `ValueError`.
- `_run_subprocess_command` / `_run_subprocess_command_with_env` (L1015/L1050): `shlex.split`
  the command, validate it, run with `shell=False`, capture output, optionally `check`.
- `_run_subprocess_command_async` (L1082): `Popen` for long-running processes (Ray start,
  Prometheus), with optional `/tmp/`-restricted stdout/stderr files and an optional post-start
  wait.

### Failure & shutdown semantics

- SIGTERM/SIGINT → `signal_handler` (L84) shuts Ray down and exits with `DEFAULT_FAILURE_CODE`
  if `has_failure` else `SUCCESS_EXIT_CODE`.
- Errors set the module global `has_failure = True` and propagate. `main` (L1859) then writes
  `/opt/ml/output/failure` via `_write_failure_reason_file` (L1848, only if the file doesn't
  already exist) and raises `RuntimeError`, ensuring **SageMaker marks the job FAILED** rather
  than silently succeeding.
- Module-level mutable globals coordinate this across functions: `ray_initialized`,
  `has_failure`, `prometheus_folder_name`.

### Key constants

| Constant                   | Value       | Meaning                        |
| -------------------------- | ----------- | ------------------------------ |
| `DEFAULT_RAY_PORT`         | 6379        | Ray GCS port                   |
| dashboard / metrics ports  | 8265 / 8080 | Ray Dashboard / metrics export |
| `RAY_CONNECTION_TIMEOUT`   | 300s        | head waits for workers         |
| `RAY_WORKER_POLL_INTERVAL` | 10s         | worker liveness poll           |
| `PROMETHEUS_WAIT_SECONDS`  | 300         | Prometheus readiness wait      |

## Conventions & gotchas

- **Edit only `scripts/launcher.py` for launcher changes.** It is the canonical copy.
- **The committed `examples/*/scripts/launcher.py` copies are STALE leftovers.** Only 5 of 8
  example dirs contain one; the 3 heterogeneous dirs (`ray-remote/pytorch-heterogeneous`,
  `ray-torchtrainer/huggingface-heterogeneous`, `ray-tune/pytorch-heterogeneous`) have none.
  The present copies diverge from canonical by ~40–256 lines. The notebooks copy
  `scripts/launcher.py` into the example `scripts/` dir at runtime, so the committed copies are
  not authoritative — do not hand-edit them to change behavior.
- **README dashboard drift (don't propagate):** the README references
  `ray_default_dashboard.json` / `ray_train_dashboard.json`, but the repo ships only
  `ray_sagemaker_training_dashboard.json`.
- **Every CLI arg has an env-var fallback** with the same name (`wait_shutdown`,
  `head_instance_group`, `head_num_cpus`, `head_num_gpus`, `launch_prometheus`,
  `prometheus_path`) via the ModelTrainer `environment` dict.
- **Prefer short env-var names** (e.g. `RAY_PROMETHEUS_USERNAME`, not
  `RAY_PROMETHEUS_REMOTE_WRITE_USERNAME`).
- **No custom Dockerfile** — runs on the stock SageMaker PyTorch container; deps come from
  `requirements.txt`.
- **Ray Autoscaler is not supported** — SageMaker cluster size is fixed at job creation.

## Common tasks

- **Change launcher behavior** → edit `scripts/launcher.py` only; ignore the example copies.
- **Add an example** → mirror an existing pattern dir: a `notebook.ipynb` plus
  `scripts/{entry script, model.py, requirements.txt}`. The notebook should copy
  `scripts/launcher.py` in and launch via the PySDK v3 `ModelTrainer` with
  `command="python launcher.py -e <script> ..."`.
- **Run an example** → open its `notebook.ipynb`; it handles data prep, launcher copy, and the
  `ModelTrainer.train()` call. See README "Example Usage" for the full snippet.

## Parameters & dependencies

Full parameter and environment-variable tables live in README → "Required Parameters and
Environment Variables" (kept there to avoid duplication/drift). Core dependency pins are in
`scripts/requirements.txt` (`ray[data,train,tune,serve]==2.54.0`, `sagemaker==3.5.0`);
per-example extras live in each example's `scripts/requirements.txt`.
