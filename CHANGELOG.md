# Changelog

## v0.1.0-beta (2026-11-03)

First public beta.

- Local chat and coding assistant running on Ollama (default model `qwen2.5-coder:14b`).
- Self-building plugins: writes a tool and a unit test, runs the test in isolation, mounts the tool only if it passes.
- Standalone command-line tool builder: describe a tool in chat and Helix writes it, its tests and a README into `workspace/`, then checks it with smoke tests, output oracles, a 50 MB memory test and its own unit tests.
- Automatic hardware tuning and `/vram` GPU memory profiles.
- Local fine-tuning pipeline (data preparation tested; the fine-tune step has not yet been run end to end).
- Licensed under the Functional Source License 1.1 (MIT future licence).
