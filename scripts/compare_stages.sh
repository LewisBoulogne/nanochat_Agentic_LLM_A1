#!/usr/bin/env bash
# Qualitative comparison of mid-trained vs SFT model on identical prompts.
set -u
PROMPTS=(
  "What is the capital of France?"
  "Which of the following is a mammal? A) Shark B) Dolphin C) Trout D) Octopus"
  "Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?"
  "Hello! Can you tell me a bit about yourself?"
  "Explain why the sky is blue."
)
for TAG in d4L-midtrain d4L-sft; do
  echo "################ $TAG ################"
  for P in "${PROMPTS[@]}"; do
    echo "---- PROMPT: $P"
    python -m scripts.chat_cli -i sft -g "$TAG" -t 0.0 -p "$P" 2>/dev/null
    echo
  done
done
