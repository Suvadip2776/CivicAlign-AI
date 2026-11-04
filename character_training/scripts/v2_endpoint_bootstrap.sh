#!/bin/sh
# Runs one training script on a rented Inference Endpoint GPU, then pauses the
# endpoint so it stops billing. Expects env: HF_TOKEN (secret), JOB_SCRIPT,
# JOB_ARGS, ENDPOINT_NAME, ENDPOINT_NAMESPACE.
#
# The endpoint health check hits port 8000; a static file server there also
# exposes the live log at /train.log (fetch with the HF token).
set -u
mkdir -p /tmp/www && cd /tmp/www
echo "booting $(date -u)" > train.log
python3 -m http.server 8000 >/dev/null 2>&1 &

{
  pip install -q uv huggingface_hub
  python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download("YOUR_HF_ORG/nycc-oct-qwen38-neutral", "scripts/" + os.environ["JOB_SCRIPT"],
                    repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir="/tmp/job")
print("fetched", p)
PY
  nvidia-smi --query-gpu=name,memory.total --format=csv
  # shellcheck disable=SC2086
  uv run /tmp/job/scripts/"$JOB_SCRIPT" $JOB_ARGS
  echo "JOB_EXIT_CODE=$?"
} >> train.log 2>&1

python3 - >> train.log 2>&1 <<'PY'
import os, time
from huggingface_hub import HfApi, pause_inference_endpoint
# Keep the log: a paused endpoint can no longer serve it.
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/tmp/www/train.log", repo_id="YOUR_HF_ORG/nycc-oct-qwen38-neutral", repo_type="dataset",
    path_in_repo=f"logs/{os.environ['ENDPOINT_NAME']}-{os.environ['JOB_SCRIPT']}-{time.strftime('%Y%m%d-%H%M%S')}.log")
pause_inference_endpoint(os.environ["ENDPOINT_NAME"], namespace=os.environ["ENDPOINT_NAMESPACE"],
                         token=os.environ["HF_TOKEN"])
print("endpoint paused")
PY
sleep infinity
