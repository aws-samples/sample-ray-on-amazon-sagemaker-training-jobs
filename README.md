# Ray on Amazon SageMaker training jobs

This repository demonstrates how to use Ray for distributed data processing and model training within Amazon SageMaker training jobs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
  - [Launcher](#launcher)
  - [Required Parameters and Environment Variables](#required-parameters-and-environment-variables)
    - [Parameter Reference](#parameter-reference)
    - [Environment Variables Reference](#environment-variables-reference)
  - [Script definition](#script-definition)
- [Examples](#examples)
- [Example Usage](#example-usage)
- [Ray Dashboard](#ray-dashboard)
- [Observability with Prometheus and Grafana](#observability-with-prometheus-and-grafana)
  - [Remote Write to an External Prometheus](#remote-write-to-an-external-prometheus)
    - [Amazon Managed Service for Prometheus (AMP)](#amazon-managed-service-for-prometheus-amp)
    - [Self-hosted Prometheus](#self-hosted-prometheus)
  - [Grafana Dashboards](#grafana-dashboards)

## Prerequisites

- AWS account with Amazon SageMaker AI access
- Ray 2.0.0+
- SageMaker Python SDK >=3.5.0

## Project Structure

```
ray-sagemaker-training/
├── scripts/
│    └── launcher.py
├── examples/
│    ├── ray-remote/
│    │    ├── pytorch/                    # Homogeneous cluster
│    │    │    ├── notebook.ipynb
│    │    │    └── scripts/
│    │    │         ├── train.py
│    │    │         ├── model.py
│    │    │         └── requirements.txt
│    │    └── pytorch-heterogeneous/      # Heterogeneous cluster
│    │         ├── notebook.ipynb
│    │         └── scripts/
│    │              ├── train.py
│    │              ├── model.py
│    │              └── requirements.txt
│    ├── ray-data/
│    │    ├── pytorch/
│    │    │    ├── notebook.ipynb
│    │    │    └── scripts/
│    │    │         ├── inference.py
│    │    │         ├── model.py
│    │    │         └── requirements.txt
│    │    └── pytorch-heterogeneous/
│    │         ├── notebook.ipynb
│    │         └── scripts/
│    │              ├── inference.py
│    │              ├── model.py
│    │              └── requirements.txt
│    ├── ray-tune/
│    │    ├── pytorch/
│    │    │    ├── notebook.ipynb
│    │    │    └── scripts/
│    │    │         ├── tune.py
│    │    │         ├── model.py
│    │    │         └── requirements.txt
│    │    └── pytorch-heterogeneous/
│    │         ├── notebook.ipynb
│    │         └── scripts/
│    │              ├── tune.py
│    │              ├── model.py
│    │              └── requirements.txt
│    └── ray-torchtrainer/
│         ├── huggingface/
│         │    ├── notebook.ipynb
│         │    └── scripts/
│         │         ├── train_ray.py
│         │         └── requirements.txt
│         └── huggingface-heterogeneous/
│              ├── notebook.ipynb
│              └── scripts/
│                   ├── train_ray.py
│                   └── requirements.txt
├── grafana-dashboards/
│    ├── ray_default_dashboard.json
│    └── ray_train_dashboard.json
└── images/
```

## Key Components

### Launcher

The `launcher.py` script serves as the entry point for SageMaker training jobs and handles:

- Setting up the Ray environment for both single-node and multi-node scenarios
- Supporting both homogeneous and heterogeneous instance group clusters
- Coordinating between head and worker nodes in a distributed setup
- Configuring EFA/RDMA networking for supported GPU instances
- Optionally launching Prometheus for metrics collection
- Executing the appropriate user script (Python `.py` or Bash `.sh`)
- Graceful shutdown with configurable wait period

#### Important Notes

**The `launcher.py` script is not intended to be modified by users.** This script serves as a universal entrypoint for SageMaker training jobs and handles Ray cluster setup, coordination between nodes, and execution of your custom scripts.

**Ray Autoscaler is not supported.** SageMaker training jobs use a fixed number of instances defined at job creation time. The Ray cluster size is determined by the SageMaker cluster configuration (`instance_count` or `instance_groups`), and cannot be dynamically scaled during execution. All nodes are provisioned at the start of the job and remain available until the job completes.

You should:

- Write your own Ray scripts for data processing or model training
- Use `launcher.py` as the entrypoint in your SageMaker jobs
- Make sure your `requirements.txt` or your container includes `ray[data,train,tune,serve]` and `sagemaker`
- Specify the custom script path using the `-e` / `--entrypoint` argument

### Required Parameters and Environment Variables

The `launcher.py` script requires specific parameters to execute your custom training scripts. You can configure these through command line arguments or environment variables.

### Parameter Reference

| Argument                | Type   | Required | Default          | Description                                                              |
| ----------------------- | ------ | -------- | ---------------- | ------------------------------------------------------------------------ |
| `-e`, `--entrypoint`    | string | Yes      | None             | Path to your script (e.g., `train.py`, `training/train.py`, `run.sh`)    |
| `--head-instance-group` | string | Yes\*    | None             | Instance group name for Ray head node (heterogeneous clusters only)      |
| `--head-num-cpus`       | int    | No       | Instance default | Number of CPUs reserved for head node                                    |
| `--head-num-gpus`       | int    | No       | Instance default | Number of GPUs reserved for head node                                    |
| `--include-dashboard`   | bool   | No       | True             | Enable Ray dashboard                                                     |
| `--launch-prometheus`   | bool   | No       | True             | Launch local Prometheus on the head node. Internet connectivity required |
| `--prometheus-path`     | string | No       | None             | Path to prometheus binary if provided as InputData                       |
| `--wait-shutdown`       | int    | No       | None             | Seconds to wait before Ray shutdown                                      |

\*Required only for heterogeneous clusters

### Environment Variables Reference

All parameters above can also be set as environment variables via the `environment` dict in your ModelTrainer or Estimator configuration. Environment variables are used as fallback when the corresponding command line argument is not provided.

| Variable                               | Type   | Required | Description                                                                                                                                                             |
| -------------------------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `head_instance_group`                  | string | No       | Alternative way to set head instance group name (heterogeneous clusters only)                                                                                           |
| `head_num_cpus`                        | int    | No       | Alternative way to set number of CPUs reserved for head node                                                                                                            |
| `head_num_gpus`                        | int    | No       | Alternative way to set number of GPUs reserved for head node                                                                                                            |
| `launch_prometheus`                    | bool   | No       | Alternative way to enable/disable local Prometheus on the head node (default: true). Internet connectivity required                                                     |
| `prometheus_path`                      | string | No       | Path to prometheus binary if provided as InputData                                                                                                                      |
| `wait_shutdown`                        | int    | No       | Alternative way to set shutdown wait time                                                                                                                               |
| `RAY_PROMETHEUS_HOST`                  | string | No       | Prometheus host URL. When set to a remote URL (not localhost), enables remote_write from local Prometheus to the remote endpoint. For AMP URLs, SigV4 auth is automatic |
| `RAY_PROMETHEUS_NAME`                  | string | No       | Prometheus data source name in Grafana (default: `Prometheus`). Used by the Ray Dashboard for Grafana integration                                                       |
| `RAY_GRAFANA_HOST`                     | string | No       | Grafana server URL. Used by Ray Dashboard for server-side API calls and browser-side iframe embedding                                                                   |
| `RAY_PROMETHEUS_REMOTE_WRITE_USERNAME` | string | No       | Username for basic auth when remote writing to a self-hosted Prometheus server                                                                                          |
| `RAY_PROMETHEUS_REMOTE_WRITE_PASSWORD` | string | No       | Password for basic auth when remote writing to a self-hosted Prometheus server                                                                                          |

### Script definition

The entry script can be a Python (`.py`) or Bash (`.sh`) file.

Python entry scripts must contain a `__main__` block:

```python
import ray

# Your Ray code here

if __name__ == "__main__":
    # This block will be executed by the launcher
    pass
```

Bash entry scripts are executed directly via `bash <script_path>`.

## Examples

The repository includes 8 example notebooks covering 4 Ray patterns, each with both homogeneous and heterogeneous cluster configurations.

Each notebook copies `launcher.py` into its local `scripts/` directory and launches a SageMaker training job using the PySDK v3 `ModelTrainer` API.

| Pattern              | Description                                                                                                                              | Homogeneous         | Heterogeneous                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------- |
| **ray-remote**       | Distributed task-level parallelism using `@ray.remote` for data cleaning and PyTorch model training (sentiment classification)           | 1x `ml.m5.2xlarge`  | 1x `ml.t3.large` (head) + 2x `ml.m5.2xlarge` (workers)  |
| **ray-data**         | Batch inference with `ray.data` using ResNet152 on Imagenette dataset                                                                    | 1x `ml.m5.2xlarge`  | 1x `ml.t3.large` (head) + 2x `ml.m5.2xlarge` (workers)  |
| **ray-tune**         | Hyperparameter tuning with `ray.tune` and ASHA scheduler on CIFAR-10                                                                     | 1x `ml.m5.2xlarge`  | 1x `ml.t3.large` (head) + 2x `ml.m5.2xlarge` (workers)  |
| **ray-torchtrainer** | Distributed LLM fine-tuning (LoRA/QLoRA) with `ray.train.torch.TorchTrainer`, HuggingFace Transformers, and optional MLflow/W&B tracking | 1x `ml.g5.12xlarge` | 1x `ml.t3.2xlarge` (head) + 4x `ml.g5.xlarge` (workers) |

In heterogeneous configurations, the head node is configured as coordinator-only (`head_num_cpus=0`, `head_num_gpus=0`), while the worker instance group handles computation.

## Example Usage

The launcher script has been designed to be flexible and dynamic, allowing you to specify any entry script through arguments or environment variables, rather than hardcoded imports.

### Entrypoint Argument

The launcher uses one argument:

- `-e` / `--entrypoint`: Path to the script to execute (Python files must contain `if __name__ == "__main__":` block)

### Usage Examples

See the content of [examples](./examples)

#### 1. Using SageMaker ModelTrainer (Recommended)

```python
from sagemaker.train.configs import (
    Compute,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

args = [
    "-e",
    "train.py",
    "--epochs",
    "25",
    "--learning_rate",
    "0.001",
    "--batch_size",
    "100",
]

# Define the source code configuration
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    command=f"python launcher.py {' '.join(args)}",
)

# Define compute configuration
compute_configs = Compute(
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    keep_alive_period_in_seconds=0,
)

# Define training job name and output path
job_name = "train-ray-training"
output_path = f"s3://{bucket_name}/{job_name}"

# Create the ModelTrainer
model_trainer = ModelTrainer(
    training_image=image_uri,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_configs,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    role=role,
)

...

# Start the training job
model_trainer.train(input_data_config=[train_input], wait=False)
```

#### 2. Heterogeneous Cluster with SageMaker ModelTrainer

```python
from sagemaker.train.configs import (
    Compute,
    InstanceGroup,
    OutputDataConfig,
    RemoteDebugConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

# Define instance groups with different instance types
instance_groups = [
    InstanceGroup(
        instance_group_name="head-instance-group",
        instance_type="ml.t3.large",       # CPU-only for coordination
        instance_count=1,
    ),
    InstanceGroup(
        instance_group_name="worker-instance-group-1",
        instance_type="ml.m5.2xlarge",     # Compute instances for training
        instance_count=2,
    ),
]

args = [
    "--entrypoint",
    "train.py",
    "--epochs",
    "100",
    "--learning_rate",
    "0.001",
    "--batch_size",
    "100",
]

# Define the source code configuration
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    command=f"python launcher.py {' '.join(args)}",
)

# Define compute with instance groups
compute_configs = Compute(
    instance_groups=instance_groups,
    keep_alive_period_in_seconds=0,
)

# Define training job name and output path
job_name = "train-ray-training"
output_path = f"s3://{bucket_name}/{job_name}"

# Create the ModelTrainer
model_trainer = ModelTrainer(
    training_image=image_uri,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_configs,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    output_data_config=OutputDataConfig(
        s3_output_path=output_path, compression_type="NONE"
    ),
    environment={
        "head_instance_group": "head-instance-group",  # Specify which group is the head
        "head_num_cpus": "0",   # Head node as coordinator only
        "head_num_gpus": "0",   # Head node as coordinator only
    },
    role=role,
).with_remote_debug_config(RemoteDebugConfig(enable_remote_debug=True))

...

# Start the training job
model_trainer.train(input_data_config=[train_input], wait=False)
```

**Key environment variables for heterogeneous clusters:**

- `head_instance_group`: Specifies which instance group should act as the Ray head node
- `head_num_cpus`: Number of CPUs to reserve for the head node (set to `"0"` for coordinator-only mode)
- `head_num_gpus`: Number of GPUs to reserve for the head node (set to `"0"` for coordinator-only mode)

### Entry Script Requirements

Your entry scripts must follow this pattern:

```python
# my_script.py
import ray

# Your Ray code here

if __name__ == "__main__":
    # This block will be executed by the launcher
    # Ray is already initialized — use ray.cluster_resources(), @ray.remote, etc.
    pass
```

## Ray Dashboard

For accessing the Ray Dashboard during the execution of Ray workload, we can leverage the native feature to access [SageMaker training jobs by using AWS System Manager (SSM)](https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-debugging.html)

### Step 1: Setup IAM Permissions:

Please refer to the official [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-debugging.html#train-remote-debugging-iam)

### Step 2:

Enable remote debugging for SageMaker training jobs:

```python
from sagemaker.train.configs import (
    CheckpointConfig,
    Compute,
    OutputDataConfig,
    RemoteDebugConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

# Define the script to be run
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    command="python launcher.py --entrypoint train_ray.py",
)

# Define the compute
compute_configs = Compute(
    instance_type=instance_type,
    instance_count=instance_count,
    keep_alive_period_in_seconds=0,
)

...

# Define the ModelTrainer
model_trainer = ModelTrainer(
    training_image=image_uri,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_configs,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    checkpoint_config=CheckpointConfig(
        s3_uri=output_path + "/checkpoint", local_path="/opt/ml/checkpoints"
    ),
    role=role,
).with_remote_debug_config(RemoteDebugConfig(enable_remote_debug=True))
```

### Step 3:

Access the training container, by starting a Port Forwarding to the port `8265` (Default Ray Dashboard port) with the following command:

```
aws ssm start-session --target sagemaker-training-job:<training-job-name>_algo-<n> \
--region <aws_region> \
--document-name AWS-StartPortForwardingSession \
--parameters '{"portNumber":["8265"],"localPortNumber":["8265"]}'
```

In a multi-node cluster, you can check the head node by investigating the CloudWatch logs:

```
2025-06-25 08:47:18,755 - __main__ - INFO - Found multiple hosts, initializing Ray as a multi-node cluster
2025-06-25 08:47:18,755 - __main__ - INFO - Head node: algo-1, Current host: algo-3
```

### Step 4:

Access the Ray Dashboard from your browser: `localhost:8265`:

![Ray Dashboard](./images/ray_dashboard.png)

## Observability with Prometheus and Grafana

Prometheus runs locally on the head node by default, collecting Ray system metrics (CPU, GPU, memory, disk, network, and Ray-specific metrics). Metrics are visible in the Ray Dashboard's metrics tab without any additional configuration.

> **Note:** Internet connectivity on the SageMaker cluster is required for Prometheus to be downloaded automatically. See [Provide Prometheus binary file](#optional-provide-prometheus-binary-file) for offline environments.

### Remote Write to an External Prometheus

You can forward metrics to an external Prometheus-compatible endpoint using `remote_write`. This is useful for persisting metrics beyond the training job lifetime or for centralized monitoring with Grafana.

Set `RAY_PROMETHEUS_HOST` to the remote Prometheus base URL. The launcher will:

1. Keep the local Prometheus running and the Ray Dashboard connected to it (`http://127.0.0.1:9090`)
2. Automatically inject a `remote_write` section into the local Prometheus configuration
3. Build the remote write URL by appending `/api/v1/remote_write` to the provided host

#### Amazon Managed Service for Prometheus (AMP)

When `RAY_PROMETHEUS_HOST` points to an AMP endpoint (`aps-workspaces.{region}.amazonaws.com`), SigV4 authentication is automatically configured using the IAM execution role attached to the SageMaker training job.

```python
model_trainer = ModelTrainer(
    ...
    source_code=source_code,
    environment={
        "RAY_PROMETHEUS_HOST": "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/ws-xxxxx",
    },
    ...
)
```

**IAM Requirements:** The SageMaker execution role must have the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "aps:RemoteWrite",
        "aps:GetSeries",
        "aps:GetLabels",
        "aps:GetMetricMetadata"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Self-hosted Prometheus

For a self-hosted Prometheus server (e.g., on EC2), set `RAY_PROMETHEUS_HOST` to the server URL. The remote server must have `--web.enable-remote-write-receiver` enabled (Prometheus v2.33+).

```python
model_trainer = ModelTrainer(
    ...
    source_code=source_code,
    environment={
        "RAY_PROMETHEUS_HOST": "https://my-prometheus-server.example.com",
    },
    ...
)
```

If the remote Prometheus requires basic authentication, pass credentials via environment variables:

```python
model_trainer = ModelTrainer(
    ...
    source_code=source_code,
    environment={
        "RAY_PROMETHEUS_HOST": "https://my-prometheus-server.example.com",
        "RAY_PROMETHEUS_REMOTE_WRITE_USERNAME": "myuser",
        "RAY_PROMETHEUS_REMOTE_WRITE_PASSWORD": "mypassword",
    },
    ...
)
```

> **Note:** Make sure the SageMaker training job has network connectivity to the remote endpoint (e.g., via VPC configuration).

### (Optional) Provide Prometheus binary file

By default, Ray downloads the Prometheus binary from the internet. In environments with limited internet connectivity, you can pre-download the binary, upload it to S3, and provide it as a training input.

> **Warning:** If the SageMaker training job does not have internet access (e.g., running in a private VPC without a NAT gateway), Prometheus will fail to download and metrics collection will not work. In this case, you **must** provide the binary file as described below.

**Step 1:** Download the Prometheus binary:

```bash
wget https://github.com/prometheus/prometheus/releases/download/v3.4.2/prometheus-3.4.2.linux-amd64.tar.gz
```

**Step 2:** Upload to S3 and configure as training input:

```python
from sagemaker.train.configs import InputData, S3DataSource

prometheus_input = InputData(
    channel_name="prometheus",
    data_source=S3DataSource(
        s3_data_type="S3Prefix",
        s3_uri="s3://<bucket>/path/to/prometheus-3.4.2.linux-amd64.tar.gz",
        s3_data_distribution_type="FullyReplicated",
    ),
)
```

**Step 3:** Pass the `--prometheus-path` argument:

```python
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    command="python launcher.py --entrypoint train_ray.py --prometheus-path /opt/ml/input/data/prometheus/prometheus-3.4.2.linux-amd64.tar.gz",
)
```

### Grafana Dashboards

This repository includes pre-built Ray Grafana dashboards in the [`grafana-dashboards/`](./grafana-dashboards) directory:

| Dashboard                    | Description                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `ray_default_dashboard.json` | Cluster overview: CPU, GPU utilization, memory, disk, network, and Ray system metrics (38 panels) |
| `ray_train_dashboard.json`   | Ray Train specific metrics for training jobs (5 panels)                                           |

#### Importing Dashboards

1. In Grafana, go to **Dashboards** → **New** → **Import**
2. Upload the JSON file from the `grafana-dashboards/` directory
3. Select your Prometheus data source when prompted
4. Click **Import**

#### Connecting Grafana to Amazon Managed Service for Prometheus

When using AMP as the metrics backend, configure your Grafana instance (Amazon Managed Grafana or self-hosted) to read from AMP:

1. Add a new **Amazon Managed Service for Prometheus** data source (or **Prometheus** with SigV4 auth)
2. Set the **Prometheus server URL** to your AMP workspace URL (e.g., `https://aps-workspaces.us-east-1.amazonaws.com/workspaces/ws-xxxxx`)
3. Configure **SigV4 authentication** with the appropriate IAM role
4. Set the **Scrape interval** to `10s` to match the Prometheus configuration
5. Click **Save & test** to verify connectivity

> **Note:** Grafana iframe embedding in the Ray Dashboard requires `allow_embedding = true` and anonymous auth in `grafana.ini`, which is only available with self-hosted Grafana. AWS Managed Grafana does not expose these settings. Use Managed Grafana dashboards directly in a separate browser tab.

## Authors

[Bruno Pistone](https://it.linkedin.com/in/bpistone) - Sr. WW Gen AI/ML Specialist Solutions Architect - Amazon SageMaker AI

[Giuseppe A. Porcelli](https://it.linkedin.com/in/giuporcelli) - Principal, ML Specialist Solutions Architect - Amazon SageMaker AI

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Faws-samples%2Fsample-ray-on-amazon-sagemaker-training-jobs&label=Hits&icon=heart-fill&color=%23198754&message=&style=flat&tz=UTC)
