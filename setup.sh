#!/bin/bash
# AutoShorts setup for macOS (Apple Silicon or Intel).
set -e

echo "== AutoShorts setup =="

if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh"; exit 1
fi

echo "-- Installing ffmpeg…"
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg

echo "-- Installing Ollama…"
if ! command -v ollama >/dev/null; then
  brew install ollama
fi

echo "-- Creating Python 3.12 virtualenv…"
PY=python3.12
command -v $PY >/dev/null || PY=python3
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "-- Pulling default LLM (qwen3:8b, ~5 GB)…"
(ollama serve >/dev/null 2>&1 &) ; sleep 3
ollama pull qwen3:8b || echo "WARN: pull failed — run 'ollama pull qwen3:8b' manually."

echo "-- Note: the Whisper model downloads automatically on first run."
echo "-- Optional: install the Telugu subtitle font (Noto Sans Telugu):"
echo "   brew install --cask font-noto-sans-telugu"
echo
echo "Done. Verify with:  source .venv/bin/activate && python -m app check"
