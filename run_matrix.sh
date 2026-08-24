#!/bin/bash
set -e
cd /home/kyan67verado/rag_app

if [ -z "$1" ]; then
  echo "Usage: bash run_matrix.sh <ZHIPU_API_KEY> [OPENAI_API_KEY]"
  exit 1
fi
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export ZHIPU_API_KEY="$1"
if [ -n "$2" ]; then export OPENAI_API_KEY="$2"; fi
PY=ragenv/bin/python

echo "==> inspect"
$PY app/evaluate_bertscore_matrix.py inspect

echo "==> ping zhipu"
$PY app/evaluate_bertscore_matrix.py ping

echo "==> gold (gpt-4o-mini)"
$PY app/evaluate_bertscore_matrix.py gold

echo "==> retrieve base"
$PY app/evaluate_bertscore_matrix.py retrieve --embedding base --top-k 3

echo "==> retrieve finetuned"
$PY app/evaluate_bertscore_matrix.py retrieve --embedding finetuned --top-k 3

echo "==> generate base/openai"
$PY app/evaluate_bertscore_matrix.py generate --embedding base --llm openai

echo "==> generate base/zhipu"
$PY app/evaluate_bertscore_matrix.py generate --embedding base --llm zhipu

echo "==> generate finetuned/openai"
$PY app/evaluate_bertscore_matrix.py generate --embedding finetuned --llm openai

echo "==> generate finetuned/zhipu"
$PY app/evaluate_bertscore_matrix.py generate --embedding finetuned --llm zhipu

echo "==> score"
$PY app/evaluate_bertscore_matrix.py score

echo "==> DONE. Lihat: data/bertscore/bertscore_matrix.json"
