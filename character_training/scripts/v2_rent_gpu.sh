#!/bin/sh
# Rent one H200 on the a Hugging Face Inference Endpoint account (post-paid; HF Jobs need
# pre-paid credit) and run one training step on it. The
# container runs scripts/endpoint_bootstrap.sh, which serves the live log at
# <url>/train.log, uploads it to the dataset repo's logs/ when done, and pauses
# the endpoint. Delete it afterwards with:
#   hf endpoints delete nycc-qwen38-train --namespace YOUR_HF_ORG --yes
#
#   scripts/rent_gpu.sh 03_train_dpo.py "--smoke"
set -eu
SCRIPT="$1"; ARGS="${2:-}"
HF=${HF:-hf}
NAME=nycc-qwen38-train
cd "$(dirname "$0")/.."
HF_TOKEN=$($HF auth token 2>/dev/null | tail -1); export HF_TOKEN
$HF upload YOUR_HF_ORG/nycc-oct-qwen38-neutral scripts scripts --include "v2_*" --type dataset --commit-message "scripts" >/dev/null
$HF endpoints delete $NAME --namespace YOUR_HF_ORG --yes >/dev/null 2>&1 || true
BOOT=$(base64 < scripts/v2_endpoint_bootstrap.sh | tr -d '\n')
$HF endpoints deploy $NAME --namespace YOUR_HF_ORG --repo Qwen/Qwen3.8-27B \
  --framework custom --engine custom --custom-image python:3.12-bookworm \
  --vendor gcp --region us-south1 --accelerator gpu --instance-type nvidia-h200 --instance-size x1 \
  --min-replica 1 --max-replica 1 --type authenticated --port 8000 --health-route / \
  --container-command "sh" --container-args "-c 'echo \$BOOT | base64 -d > /tmp/b.sh && sh /tmp/b.sh'" \
  --env "BOOT=$BOOT" --env "JOB_SCRIPT=$SCRIPT" --env "JOB_ARGS=$ARGS" \
  --env "ENDPOINT_NAME=$NAME" --env "ENDPOINT_NAMESPACE=YOUR_HF_ORG" --secrets HF_TOKEN --format json \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["status"]; print(s["state"], s["url"])'
