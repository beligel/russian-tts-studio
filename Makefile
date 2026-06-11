.PHONY: help install test run-eval run-compare run-pipeline discover discover-speakers export-speaker synth-with-speaker import-speakers install-plugin web desktop start clean

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "XTTS pipeline for Russian TTS — make targets:"
	@echo "  install          Install all dependencies"
	@echo "  test             Run unit tests"
	@echo "  run-compare      Compare Silero and XTTS on Russian phrases"
	@echo "  run-pipeline     Run production pipeline on a text"
	@echo "  discover         Discover ComfyUI + plugin installation"
	@echo "  discover-speakers  List ComfyUI speaker presets"
	@echo "  install-plugin   Install ComfyUI plugin"
	@echo "  web              Launch the Web UI on http://localhost:8129 (HTTP only)"
	@echo "  desktop          Launch the Web UI in a native WebView window (pywebview)"
	@echo "  start            Smart launch — auto-picks native window or browser (recommended, port 8129)"
	@echo "  clean            Remove generated samples and cache"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

run-compare:
	$(PYTHON) scripts/comparison/compare_engines.py \
		--reference output/reference/ru_voice.wav \
		--engines silero,xtts \
		--output-dir output/comparison

run-pipeline:
	$(PYTHON) scripts/inference/run_pipeline.py \
		--text "Привет, это тестовая фраза для проверки пайплайна." \
		--reference output/reference/ru_voice.wav \
		--output output/samples/test.wav

discover:
	$(PYTHON) scripts/integration/comfyui_bridge.py --discover

discover-speakers:
	$(PYTHON) scripts/integration/comfyui_bridge.py --list-speakers

install-plugin:
	$(PYTHON) scripts/integration/comfyui_bridge.py --install

web:
	$(PYTHON) -m uvicorn web.app:app --host 0.0.0.0 --port 8129 --reload

desktop:
	$(PYTHON) -m web.desktop

start:
	$(PYTHON) -m web.start $(START_ARGS)

export-speaker:
	$(PYTHON) scripts/integration/comfyui_bridge.py \
		--export-speaker --audio $(AUDIO) --name $(NAME)

synth-with-speaker:
	$(PYTHON) scripts/integration/comfyui_bridge.py \
		--synthesize --text "$(TEXT)" --speaker $(SPEAKER) --output $(OUTPUT)

import-speakers:
	$(PYTHON) scripts/integration/comfyui_bridge.py --import-speakers

clean:
	rm -rf output/samples/*.wav output/comparison/**/*.wav output/reports/*.json output/reports/*.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
