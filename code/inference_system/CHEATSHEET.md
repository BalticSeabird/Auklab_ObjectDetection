# Main Orchestrator Cheatsheet

## Quick Start
```bash
cd /home/jonas/Documents/vscode/Auklab_OD
python -m code.inference_system.main_orchestrator --config config/system_config.yaml
```

## How It Works
- **All paths and model configs are defined in the YAML config file**, not as CLI arguments
- The script loads the config, then orchestrates 4 processing stages
- Specify different configs for different stations or experiments

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | Path | `config/system_config.yaml` | Path to YAML config file (defines models, paths, parameters) |
| `--stations` | List | None | Process only specific stations (space-separated: `--stations FAR3 TRI3`) |
| `--resume` | Flag | False | Resume from existing state; resets stuck/failed jobs |
| `--discover-only` | Flag | False | Only discover videos and exit (don't process) |
| `--skip-discovery` | Flag | False | Skip video discovery; use existing registry |
| `--log-level` | Choice | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--stuck-timeout` | Int | 3600 | Seconds before in-progress jobs considered stuck (with `--resume`) |

## Common Usage Patterns

### Full processing with a specific config
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml
```

### Resume interrupted run (reset stuck jobs from last hour)
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml --resume --stuck-timeout 3600
```

### Process specific stations only
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml --stations FAR3 TRI3
```

### Discover videos only (no processing)
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml --discover-only
```

### Debug mode with verbose logging
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml --log-level DEBUG
```

### Skip video discovery and use cached registry
```bash
python -m code.inference_system.main_orchestrator --config config/system_config.yaml --skip-discovery
```

## Config File Structure (Partial Reference)

Key sections in your YAML config:

```yaml
paths:
  video_roots: [list of video directories]           # Where to find input videos
  inference_output: path/to/output                   # Stage 1 output
  event_analysis_output: path/to/events              # Stage 2 output
  clips_output: path/to/clips                        # Stage 3 output
  detection_model: path/to/model.pt                  # YOLO model for Stage 1
  state_db: data/processing_state.db                 # Processing state database
  log_dir: logs                                      # Log output directory

stage4_post_classification:
  use_model: true                                    # Enable Stage 4 classifier
  model_path: models/stage4/model.json               # Stage 4 classifier model
  model_threshold: 0.49                              # Stage 4 detection threshold

priorities:
  stations: [FAR3]                                   # Which stations to prioritize
  years: [2025, 2024, 2023]                          # Which years to prioritize

filters:
  date_range:
    start: '2025-05-01'
    end: '2025-07-10'
  ignored_stations: [OVERVIEW, TRI1, ...]           # Stations to skip entirely
```

## Available Configs
- `config/system_config.yaml` - Main production config
- `config/config_FAR3.yaml` - FAR3 station only
- `config/config_TRI3.yaml` - TRI3 station only
- `config/config_stage3_branch.yaml` - Experimental Stage 3 config
- `config/config_active_learning_validation_smoke.yaml` - Testing config

## Output Locations (as configured)
- **Logs**: `config.paths.log_dir` (typically `logs/main.log`)
- **Stage 1 detections**: `config.paths.inference_output`
- **Stage 2 events**: `config.paths.event_analysis_output`
- **Stage 3 clips**: `config.paths.clips_output`
- **State database**: `config.paths.state_db`

## Troubleshooting

**Script is slow?**
- Check `hardware.gpus` config for proper GPU allocation
- Adjust batch_size in `processing.stage1_video_inference`

**Stuck jobs on restart?**
- Use `--resume --stuck-timeout 300` to reset jobs stuck >5 minutes

**Only want to process subset?**
- Use `--stations FAR3 TRI3` to limit to specific cameras
- Or edit `priorities.stations` in config file directly
