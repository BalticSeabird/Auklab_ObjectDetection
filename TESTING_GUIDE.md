# Quick Start: Inference System Smoke Test

Use this checklist to configure and run the new multi-stage pipeline end-to-end.

## 1. Prepare Configuration
1. Generate a starter config (optional):
   ```bash
   PYTHONPATH=code python -m inference_system.config_manager --generate-default --output config/system_config.yaml
   ```
2. Edit `config/system_config.yaml`:
   - `paths.video_roots`: list every mount / folder that contains station video files.
   - `paths.inference_output`, `paths.event_analysis_output`, `paths.clips_output`: writable destinations for each stage.
   - `paths.state_db`, `paths.log_dir`: local fast storage is ideal.
   - `processing.stage1_*`: frame skip, batch size, confidence threshold.
   - `processing.stage2_*`: FPS, smoothing, confidence threshold.
   - `processing.stage3_*`: clip padding, event filters, compression settings.
   - `priorities.years / stations`: order targets by urgency.
   - `filters.date_range`: for first tests, restrict to a single day.
   - `hardware.gpus.device_ids` + `hardware.cpus.worker_count`: match available resources.

## 2. Discover Videos Only
```bash
PYTHONPATH=code python -m inference_system.main_orchestrator --config config/system_config.yaml --discover-only
```
- Populates `data/processing_state.db` and logs summary to `logs/`.
- Inspect via `sqlite3 data/processing_state.db "select count(*) from videos;"` (optional).

## 3. Run a Focused Smoke Test
```bash
PYTHONPATH=code python -m inference_system.main_orchestrator --config config/system_config.yaml --stations TRI3
```
Tips:
- Keep the config date filter narrow; targeted station(s) avoid long queues.
- Press `Ctrl+C` to stop. Use `--resume` later to continue.

## 4. Monitor Progress
- Console prints periodic summaries (completed / pending / failed).
- Detailed worker logs: `logs/workers/gpu0-stage1.log`, `logs/workers/cpu0-stage2.log`, etc.
- Outputs appear at each stage:
  - Stage 1 CSVs: `{inference_output}/{year}/{model}/{station}/...`
  - Stage 2 summaries: `{event_analysis_output}/{year}/{model}/{station}/{date}/`
  - Stage 3 clips: `{clips_output}/{station}/{date}/...`

## 5. Iterate / Scale Up
- Relax `filters.date_range`, remove `--stations`, or increase worker counts once the smoke test succeeds.
- Re-run `--discover-only` periodically to capture new videos.
- For restarts after interruptions: `PYTHONPATH=code python -m inference_system.main_orchestrator --config config/system_config.yaml --resume`.
