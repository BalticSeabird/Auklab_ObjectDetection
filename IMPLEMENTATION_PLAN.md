# Implementation Plan: High-Performance Video Inference System

## Executive Summary

This document provides a detailed, modular implementation plan for creating an integrated video inference system that processes multiple videos using YOLO object detection, identifies ecological events, and creates video clips for groundtruthing. The system will efficiently utilize available hardware (2 GPUs, 128 CPUs), support resumable execution, and provide comprehensive monitoring.

---

## 1. CURRENT STATE ANALYSIS

### 1.1 Existing Scripts Analysis

#### **run_inference_nth_decode.py**
- **Purpose**: Run YOLO object detection on videos
- **Current Capabilities**:
  - Batch video processing with PyAV
  - GPU-accelerated inference
  - Frame skipping (processes every 25th frame)
  - CSV output with detection results
  - Basic date range filtering
  - File existence checking to avoid reprocessing
- **Limitations**:
  - Hardcoded paths and parameters
  - Single station per execution
  - Manual device selection via command-line
  - Limited error handling
  - No state persistence
  - Single GPU usage per invocation
  - Email notification system (currently disabled)

#### **batch_analyze_days.py**
- **Purpose**: Post-process detections to identify ecological events
- **Current Capabilities**:
  - Multi-station batch processing
  - Date-based file grouping
  - Event detection (arrivals/departures)
  - Fish association with arrivals
  - Flapping event detection
  - Movement metric computation
  - Daily summary reports and visualizations
  - Skip-already-processed logic
- **Strengths**:
  - Well-structured modular functions
  - Comprehensive event analysis
  - Good visualization capabilities
  - Multi-station support with automatic path construction

#### **extract_event_clips.py**
- **Purpose**: Create video clips around detected events
- **Current Capabilities**:
  - Event-based clip extraction using ffmpeg
  - YOLO re-detection on clips with bounding boxes
  - Automatic path construction for multi-station processing
  - Video compression with H.264
  - Event categorization (arrival with/without fish, departure)
  - Skip-already-processed logic
  - CSV output of clip-level detections
- **Strengths**:
  - Robust video file finding across date subdirectories
  - Text overlay on clips
  - Organized output structure
  - Good error handling for missing videos

### 1.2 Integration Opportunities

The three scripts represent a sequential pipeline:
1. **Video Inference** (run_inference_nth_decode.py) → Raw detection CSVs
2. **Event Detection** (batch_analyze_days.py) → Event CSVs and summaries
3. **Clip Extraction** (extract_event_clips.py) → Video clips with annotations

Current gaps for integrated system:
- No unified orchestration
- No shared state/progress tracking
- No parallel multi-GPU utilization
- No centralized configuration
- No comprehensive logging system
- No priority-based scheduling
- Limited resume capability

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  - Main Controller                                           │
│  - Configuration Management                                  │
│  - Job Scheduler with Priority Queue                         │
│  - State Manager (Resume/Stop/Start)                         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌──────▼────────┐
│  GPU Worker    │  │  GPU Worker   │  │  CPU Worker   │
│  Pool (2x)     │  │  Pool (2x)    │  │  Pool (Nx)    │
│  Stage 1       │  │  Stage 3      │  │  Stage 2      │
└───────┬────────┘  └──────┬────────┘  └──────┬────────┘
        │                   │                   │
┌───────▼────────────────────▼───────────────────▼────────┐
│              PROCESSING LAYER                            │
│  Stage 1: Video Inference (GPU)                          │
│  Stage 2: Event Detection (CPU)                          │
│  Stage 3: Clip Extraction + YOLO (GPU)                   │
└──────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌──────▼────────┐
│  State DB      │  │  Log System   │  │  Monitoring   │
│  (SQLite/JSON) │  │  (Structured) │  │  (Metrics)    │
└────────────────┘  └───────────────┘  └───────────────┘
```

### 2.2 Core Components

#### **2.2.1 Configuration Manager**
- **File Format**: YAML configuration file
- **Responsibilities**:
  - Load/validate configuration
  - Manage paths (videos, models, outputs)
  - Store processing parameters
  - Define station priorities
  - Hardware resource allocation
  
#### **2.2.2 State Manager**
- **Storage**: SQLite database or JSON file
- **Tracks**:
  - Video processing status (pending/in-progress/completed/failed)
  - Station progress
  - Processing timestamps
  - Error information
  - Pipeline stage completion per video
- **Functions**:
  - Mark video as in-progress
  - Mark video as completed
  - Mark video as failed with error details
  - Query unprocessed videos
  - Support resume from last state
  
#### **2.2.3 Job Scheduler**
- **Responsibilities**:
  - Video discovery based on folder structure
  - Priority-based ordering (year/station priority from config)
  - Job queue management
  - GPU/CPU resource allocation
  - Prevent duplicate processing
  
#### **2.2.4 Worker Pool Manager**
- **GPU Workers**: Fixed pool (2 workers)
  - Stage 1: Video inference workers
  - Stage 3: Clip extraction with YOLO workers
- **CPU Workers**: Configurable pool size
  - Stage 2: Event detection workers
- **Responsibilities**:
  - Worker lifecycle management
  - Job assignment
  - Error handling and retry logic
  - Resource monitoring
  
#### **2.2.5 Logging System**
- **Hierarchical Logging**:
  - Master log: System-level events
  - Worker logs: Per-worker progress
  - Error log: All failures with stack traces
  - Performance log: Processing metrics
- **Log Rotation**: Size-based rotation
- **Format**: Structured logging with timestamps, levels, context

#### **2.2.6 Monitoring Dashboard (Optional)**
- **Metrics Tracked**:
  - Videos processed/remaining
  - Processing rate (videos/hour)
  - GPU utilization
  - Current jobs per worker
  - ETA for completion
  - Error rate
- **Output**: Log file with periodic summaries (extensible to web dashboard)

---

## 3. DETAILED MODULE SPECIFICATIONS

### 3.1 Module: Configuration Manager (`config_manager.py`)

#### **Purpose**
Centralized configuration management for all system parameters.

#### **Key Classes**

**`ConfigManager`**
- Methods:
  - `load_config(config_path: str) -> Config`
  - `validate_config(config: Config) -> bool`
  - `get_station_priority() -> List[str]`
  - `get_processing_params() -> ProcessingParams`

#### **Configuration Schema** (YAML)

```yaml
# System Configuration
system:
  name: "Auklab Video Inference System"
  version: "1.0"
  
# Hardware Resources
hardware:
  gpus:
    count: 2
    device_ids: [0, 1]
  cpus:
    worker_count: 16  # Adjust based on workload
  
# Paths
paths:
  # Input paths
  video_base: "/mnt/BSP_NAS2_vol4/Video"
  
  # Output paths
  inference_output: "/mnt/BSP_NAS2_work/auklab_model/inference"
  event_analysis_output: "/mnt/BSP_NAS2_work/auklab_model/summarized_inference"
  clips_output: "/mnt/BSP_NAS2_work/auklab_model/event_data"
  
  # Model paths
  detection_model: "models/auklab_model_xlarge_combined_6080_v1.pt"
  
  # System paths
  state_db: "data/processing_state.db"
  log_dir: "logs/"
  
# Processing Parameters
processing:
  stage1_video_inference:
    frame_skip: 25
    batch_size: 32
    confidence_threshold: 0.25
    
  stage2_event_detection:
    fps: 1
    original_video_fps: 25
    smooth_window_s: 3
    error_window_s: 10
    hold_seconds: 8
    fish_window_s: 5
    movement_smoothing_s: 5
    flap_area_multiplier: 3.0
    flap_baseline_s: 30
    
  stage3_clip_extraction:
    clip_before: 5
    clip_after: 10
    event_types: ["arrival", "departure"]  # or subset
    fish_only: false
    video_extensions: [".mkv", ".mp4", ".avi"]
    compression:
      enabled: true
      crf: 28
      preset: "fast"

# Station and Year Priorities
priorities:
  years: [2025, 2024, 2023]  # Process in this order
  stations:  # Process in this order within each year
    - "BONDEN3"
    - "BONDEN6"
    - "TRI3"
    - "FAR3"
    - "FAR6"
    - "ROST2"
    - "ROST6"

# Date Range Filters (optional)
filters:
  date_range:
    start: "2025-01-01"  # null for no limit
    end: "2025-12-31"    # null for no limit

# Retry and Error Handling
error_handling:
  max_retries: 2
  retry_delay_seconds: 60
  skip_on_persistent_failure: true
  
# Monitoring
monitoring:
  update_interval_seconds: 60
  log_performance_metrics: true
  
# Resume Settings
resume:
  enabled: true
  checkpoint_interval_seconds: 300  # Save state every 5 minutes
```

#### **Implementation Notes**
- Use `PyYAML` or `ruamel.yaml` for parsing
- Implement validation using `pydantic` or similar
- Support environment variable substitution in paths
- Provide config file template generator

---

### 3.2 Module: State Manager (`state_manager.py`)

#### **Purpose**
Track processing state for all videos to enable resume, skip completed work, and monitor progress.

#### **Key Classes**

**`StateManager`**
- Methods:
  - `initialize_db()`
  - `discover_videos(video_paths: List[Path]) -> List[VideoJob]`
  - `get_pending_jobs(stage: int) -> List[VideoJob]`
  - `mark_job_started(video_id: str, stage: int, worker_id: str)`
  - `mark_job_completed(video_id: str, stage: int, metadata: dict)`
  - `mark_job_failed(video_id: str, stage: int, error: str, retry_count: int)`
  - `get_progress_summary() -> ProgressSummary`
  - `reset_stuck_jobs(timeout_seconds: int)`
  
#### **Database Schema** (SQLite)

```sql
-- Videos table
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    station TEXT NOT NULL,
    year INTEGER NOT NULL,
    date TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    priority_score REAL NOT NULL,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing stages table
CREATE TABLE processing_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    stage INTEGER NOT NULL,  -- 1: inference, 2: events, 3: clips
    status TEXT NOT NULL,  -- pending, in_progress, completed, failed, skipped
    worker_id TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON with stage-specific info
    FOREIGN KEY (video_id) REFERENCES videos(video_id),
    UNIQUE(video_id, stage)
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    stage INTEGER NOT NULL,
    video_duration_seconds REAL,
    processing_duration_seconds REAL,
    fps_processed REAL,
    detections_count INTEGER,
    events_count INTEGER,
    clips_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

-- System state table (for resume)
CREATE TABLE system_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **Implementation Notes**
- Use `sqlite3` (built-in) or `sqlalchemy` for database operations
- Implement connection pooling for multi-threaded access
- Add indexes on frequently queried columns (status, station, priority_score)
- Provide methods to export state to JSON for backup
- Implement automatic cleanup of orphaned "in_progress" jobs on startup

---

### 3.3 Module: Job Scheduler (`job_scheduler.py`)

#### **Purpose**
Discover videos, assign priorities, and create job queues for each pipeline stage.

#### **Key Classes**

**`JobScheduler`**
- Methods:
  - `discover_videos() -> int`
  - `calculate_priorities()`
  - `build_job_queue(stage: int) -> queue.PriorityQueue`
  - `get_next_job(stage: int, gpu_id: Optional[int] = None) -> Optional[VideoJob]`
  - `return_job(job: VideoJob, stage: int)`  # For failed jobs to retry
  
**`VideoJob`** (dataclass)
- Attributes:
  - `video_id: str`
  - `station: str`
  - `year: int`
  - `date: str`
  - `filepath: Path`
  - `priority_score: float`
  - `stage: int`
  - `retry_count: int`

#### **Priority Calculation Algorithm**

```python
def calculate_priority(video: VideoJob, config: Config) -> float:
    """
    Priority score = year_weight * 1000 + station_weight * 10 + date_weight
    
    Higher score = higher priority (processed sooner)
    """
    year_priorities = {year: len(config.priorities.years) - idx 
                      for idx, year in enumerate(config.priorities.years)}
    station_priorities = {station: len(config.priorities.stations) - idx
                         for idx, station in enumerate(config.priorities.stations)}
    
    year_weight = year_priorities.get(video.year, 0)
    station_weight = station_priorities.get(video.station, 0)
    
    # Date weight: earlier dates have slightly higher priority
    date_obj = datetime.strptime(video.date, '%Y-%m-%d')
    days_from_year_start = (date_obj - datetime(video.year, 1, 1)).days
    date_weight = 365 - days_from_year_start  # 0-365
    
    return year_weight * 1000 + station_weight * 10 + date_weight / 365
```

#### **Video Discovery Process**

1. Scan video base directories: `{video_base}/Video{year}/{station}/`
2. Find all video files matching extensions (`.mkv`, `.mp4`, `.avi`)
3. Exclude files with `@eaDir` in path (NAS metadata)
4. Parse filename to extract date/time
5. Check against date range filters
6. Check if already in database, add if new
7. Calculate priority score
8. Return count of discovered videos

#### **Implementation Notes**
- Use `pathlib` for file system operations
- Implement filename parsing for both standard and XProtect formats
- Use `priority queue` from `queue` module or `heapq`
- Support incremental discovery (can re-run to find new videos)
- Handle videos that can't be parsed (log as skipped)

---

### 3.4 Module: Worker Pool Manager (`worker_pool.py`)

#### **Purpose**
Manage pools of worker processes/threads for parallel execution across pipeline stages.

#### **Key Classes**

**`WorkerPoolManager`**
- Methods:
  - `start_workers()`
  - `stop_workers(graceful: bool = True)`
  - `monitor_workers()`
  - `restart_failed_worker(worker_id: str)`

**`GPUWorker`**
- Attributes:
  - `worker_id: str`
  - `gpu_id: int`
  - `stage: int` (1 or 3)
  - `process: multiprocessing.Process`
- Methods:
  - `run()`
  - `process_job(job: VideoJob)`
  - `shutdown()`

**`CPUWorker`**
- Attributes:
  - `worker_id: str`
  - `stage: int` (2)
  - `process: multiprocessing.Process`
- Methods:
  - `run()`
  - `process_job(job: VideoJob)`
  - `shutdown()`

#### **Worker Lifecycle**

1. **Initialization**: Create worker processes with assigned resources (GPU ID for GPU workers)
2. **Job Loop**:
   - Request next job from scheduler
   - Mark job as in-progress in state manager
   - Execute processing function
   - On success: Mark completed, save results
   - On failure: Mark failed, log error, increment retry count
3. **Shutdown**: Drain current job, cleanup resources

#### **Resource Allocation**

```python
# Stage 1: Video Inference (GPU)
gpu_workers_stage1 = [
    GPUWorker(worker_id="gpu0-stage1", gpu_id=0, stage=1),
    GPUWorker(worker_id="gpu1-stage1", gpu_id=1, stage=1)
]

# Stage 2: Event Detection (CPU)
cpu_workers_stage2 = [
    CPUWorker(worker_id=f"cpu{i}-stage2", stage=2)
    for i in range(config.hardware.cpus.worker_count)
]

# Stage 3: Clip Extraction (GPU)
gpu_workers_stage3 = [
    GPUWorker(worker_id="gpu0-stage3", gpu_id=0, stage=3),
    GPUWorker(worker_id="gpu1-stage3", gpu_id=1, stage=3)
]
```

#### **Pipeline Coordination**

- Stage 2 workers wait for Stage 1 output (event-driven or polling)
- Stage 3 workers wait for Stage 2 output
- Use inter-process communication (queues, signals) for coordination

#### **Implementation Notes**
- Use `multiprocessing.Process` for true parallelism (avoid GIL)
- Implement proper signal handling (SIGTERM, SIGINT) for graceful shutdown
- Use process-safe logging (separate log files per worker or queue-based logging)
- Monitor worker health (heartbeat mechanism)
- Implement watchdog to restart crashed workers

---

### 3.5 Module: Pipeline Stages (Refactored Processing Logic)

#### **3.5.1 Stage 1: Video Inference** (`stage1_inference.py`)

**Purpose**: Run YOLO detection on videos

**Refactoring from `run_inference_nth_decode.py`:**

**`VideoInferenceProcessor`**
- Methods:
  - `process_video(video_path: Path, output_dir: Path, config: ProcessingConfig) -> InferenceResult`
  - `_load_model(model_path: Path, gpu_id: int) -> YOLO`
  - `_batch_inference(video_path: Path, model: YOLO, config: ProcessingConfig) -> pd.DataFrame`
  - `_save_detections(detections: pd.DataFrame, output_path: Path)`

**Key Changes:**
- Accept configuration object instead of hardcoded values
- Return structured result object with metadata
- Improve error handling (corrupted videos, codec issues)
- Add progress callback for monitoring
- Support graceful interruption

**Output:**
- CSV file: `{output_dir}/{station}/{filename}_raw.csv`
- Metadata: video duration, frames processed, detection count

#### **3.5.2 Stage 2: Event Detection** (`stage2_events.py`)

**Purpose**: Analyze detection CSVs to identify ecological events

**Refactoring from `batch_analyze_days.py`:**

**`EventDetectionProcessor`**
- Methods:
  - `process_detection_csv(csv_path: Path, output_dir: Path, config: ProcessingConfig) -> EventResult`
  - `_load_and_aggregate(csv_path: Path) -> Tuple[DataFrame, ...]`
  - `_detect_events(...) -> List[Event]`
  - `_generate_summaries(events: List[Event], output_dir: Path)`

**Key Changes:**
- Process single date at a time (called by worker for each video's date)
- Accept configuration object
- Return structured result with event counts
- Maintain existing analysis capabilities
- Skip if output already exists (idempotent)

**Output:**
- Daily summary directory: `{output_dir}/{station}/{date}/`
  - `csv/daily_events_{date}.csv`
  - `csv/daily_flaps_{date}.csv`
  - `plots/daily_overview_{date}.png`
  - `daily_summary_{date}.txt`

#### **3.5.3 Stage 3: Clip Extraction** (`stage3_clips.py`)

**Purpose**: Create video clips for events with YOLO overlay

**Refactoring from `extract_event_clips.py`:**

**`ClipExtractionProcessor`**
- Methods:
  - `process_event_csv(event_csv: Path, video_dir: Path, output_dir: Path, config: ProcessingConfig) -> ClipResult`
  - `_find_source_video(timestamp: datetime, video_dir: Path) -> Optional[Path]`
  - `_extract_clip(...) -> bool`
  - `_run_yolo_on_clip(...) -> bool`

**Key Changes:**
- Accept configuration object
- Return structured result with clip counts
- Maintain existing capabilities (YOLO overlay, compression)
- Skip if clips already exist (idempotent)

**Output:**
- Clip directory: `{output_dir}/{station}/{date}/{event_type}/(video|csv)/`
  - `video/{event_id}.mp4`
  - `csv/{event_id}_detections.csv`

#### **Implementation Notes for All Stages**
- Each stage should be idempotent (safe to re-run)
- Check for existing outputs before processing
- Use structured exception types for different failure modes
- Log processing start/end with timing information
- Clean up temporary files on failure

---

### 3.6 Module: Logging System (`logging_config.py`)

#### **Purpose**
Comprehensive, structured logging across all components.

#### **Log File Structure**

```
logs/
├── master.log              # Main orchestrator logs
├── scheduler.log           # Job scheduling decisions
├── workers/
│   ├── gpu0-stage1.log
│   ├── gpu1-stage1.log
│   ├── cpu0-stage2.log
│   ├── ...
│   └── gpu1-stage3.log
├── errors.log              # All errors consolidated
├── performance.log         # Processing metrics
└── monitoring.log          # System status updates
```

#### **Log Format**

```
[TIMESTAMP] [LEVEL] [COMPONENT] [CONTEXT] Message
[2026-01-15 14:32:15] [INFO] [gpu0-stage1] [video_id=TRI3_20250630T122345] Starting inference
[2026-01-15 14:32:45] [INFO] [gpu0-stage1] [video_id=TRI3_20250630T122345] Completed: 1200 detections in 30.2s
```

#### **Implementation**

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: Path, component: str) -> logging.Logger:
    """Setup logger for a component"""
    logger = logging.getLogger(component)
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    log_file = log_dir / f"{component}.log"
    handler = RotatingFileHandler(
        log_file, 
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Console handler for errors
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger
```

#### **Key Logging Points**
- System startup/shutdown
- Configuration loading
- Video discovery results
- Job scheduling decisions
- Worker start/stop
- Job start/completion/failure
- Performance metrics
- Error stack traces
- Progress summaries

---

### 3.7 Module: Main Orchestrator (`main_orchestrator.py`)

#### **Purpose**
Entry point that coordinates all components.

#### **Key Functions**

**`main()`**
```python
def main():
    """Main orchestration function"""
    # 1. Load configuration
    config = load_config("config/system_config.yaml")
    
    # 2. Setup logging
    setup_logging(config.paths.log_dir, "orchestrator")
    logger = logging.getLogger("orchestrator")
    logger.info("Starting Auklab Video Inference System")
    
    # 3. Initialize state manager
    state_mgr = StateManager(config.paths.state_db)
    state_mgr.initialize_db()
    
    # 4. Check for resume
    if config.resume.enabled:
        logger.info("Resume enabled: checking for incomplete jobs")
        state_mgr.reset_stuck_jobs(timeout_seconds=3600)
    
    # 5. Discover videos
    scheduler = JobScheduler(config, state_mgr)
    num_videos = scheduler.discover_videos()
    logger.info(f"Discovered {num_videos} videos")
    
    # 6. Calculate priorities
    scheduler.calculate_priorities()
    
    # 7. Start worker pools
    worker_mgr = WorkerPoolManager(config, state_mgr, scheduler)
    worker_mgr.start_workers()
    
    # 8. Start monitoring thread
    monitor = SystemMonitor(config, state_mgr)
    monitor.start()
    
    # 9. Wait for completion or interruption
    try:
        while not state_mgr.is_all_complete():
            time.sleep(10)
            # Periodic checkpoint
            if time.time() % config.resume.checkpoint_interval_seconds < 10:
                state_mgr.save_checkpoint()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    
    # 10. Graceful shutdown
    logger.info("Initiating graceful shutdown")
    worker_mgr.stop_workers(graceful=True)
    monitor.stop()
    
    # 11. Final report
    summary = state_mgr.get_progress_summary()
    logger.info(f"Processing complete: {summary}")
```

**`resume()`**
```python
def resume():
    """Resume from previous state"""
    # Similar to main() but skip video discovery
    # Continue with existing state
    pass
```

#### **Command-Line Interface**

```bash
# Start fresh processing
python main_orchestrator.py --config config/system_config.yaml

# Resume from last state
python main_orchestrator.py --config config/system_config.yaml --resume

# Process specific stations only
python main_orchestrator.py --config config/system_config.yaml --stations TRI3 BONDEN6

# Dry run (discover videos, calculate priorities, but don't process)
python main_orchestrator.py --config config/system_config.yaml --dry-run

# Reset state for specific stations (reprocess everything)
python main_orchestrator.py --config config/system_config.yaml --reset-stations TRI3
```

#### **Implementation Notes**
- Use `argparse` for CLI
- Implement proper signal handling (SIGTERM, SIGINT, SIGHUP)
- Support running as daemon/background service
- Create PID file to prevent multiple instances
- Log startup configuration summary

---

### 3.8 Module: Monitoring System (`monitoring.py`)

#### **Purpose**
Track and report system status, progress, and performance.

#### **Key Classes**

**`SystemMonitor`**
- Methods:
  - `start()`
  - `stop()`
  - `generate_status_report() -> StatusReport`
  - `log_performance_metrics()`

**`StatusReport`** (dataclass)
```python
@dataclass
class StatusReport:
    timestamp: datetime
    
    # Overall progress
    total_videos: int
    completed_videos: int
    failed_videos: int
    in_progress_videos: int
    pending_videos: int
    
    # Stage breakdown
    stage1_completed: int
    stage2_completed: int
    stage3_completed: int
    
    # Performance
    videos_per_hour: float
    avg_processing_time_seconds: float
    
    # ETA
    estimated_completion: datetime
    
    # Current activity
    active_workers: List[WorkerStatus]
    
    # Errors
    recent_errors: List[ErrorSummary]
```

#### **Status Report Format** (logged every 60 seconds)

```
================================================================================
SYSTEM STATUS REPORT - 2026-01-15 14:35:00
================================================================================
PROGRESS:
  Total Videos: 12,450
  Completed: 8,320 (66.8%)
  In Progress: 18 (GPU: 4, CPU: 14)
  Pending: 4,112 (33.2%)
  Failed: 0

STAGE BREAKDOWN:
  Stage 1 (Inference): 8,320 / 12,450 (66.8%)
  Stage 2 (Events): 8,305 / 8,320 (99.8%)
  Stage 3 (Clips): 8,290 / 8,305 (99.8%)

PERFORMANCE:
  Videos/Hour: 245.3
  Avg Processing Time: 14.7 seconds per video
  ETA: 2026-01-15 21:15:00 (6h 40m remaining)

ACTIVE WORKERS:
  gpu0-stage1: Processing TRI3_20250630T122345 (25% complete)
  gpu1-stage1: Processing BONDEN6_20250701T093015 (60% complete)
  cpu5-stage2: Processing FAR3_20250628T140532
  ...

TOP STATIONS:
  TRI3: 2,340 / 3,200 (73.1%)
  BONDEN6: 1,850 / 2,100 (88.1%)
  FAR3: 1,420 / 1,900 (74.7%)

RECENT ERRORS (last hour): 0
================================================================================
```

#### **Performance Metrics to Track**
- Videos processed per hour (overall and per worker)
- Average processing time per stage
- GPU utilization (if nvidia-smi available)
- Disk I/O (read/write speeds)
- Error rate
- Retry statistics

#### **Implementation Notes**
- Run in separate thread
- Use `threading.Timer` for periodic updates
- Calculate ETA based on recent processing rate (moving average)
- Export metrics to JSON for external monitoring tools
- Optional: Implement Prometheus metrics endpoint

---

## 4. DATA FLOW

### 4.1 End-to-End Pipeline Flow

```
1. VIDEO DISCOVERY
   └─> Scan video directories
   └─> Parse filenames
   └─> Apply filters (date range)
   └─> Calculate priorities
   └─> Insert into state database

2. STAGE 1: INFERENCE (GPU)
   └─> Get next high-priority video
   └─> Load YOLO model on GPU
   └─> Batch process frames (every 25th)
   └─> Save detections CSV
   └─> Update state: Stage 1 complete
   └─> Log performance metrics

3. STAGE 2: EVENT DETECTION (CPU)
   └─> Wait for Stage 1 output
   └─> Load detection CSV for date
   └─> Aggregate detections per second
   └─> Detect arrivals/departures
   └─> Associate fish with events
   └─> Detect flapping events
   └─> Generate summaries and plots
   └─> Save event CSV
   └─> Update state: Stage 2 complete

4. STAGE 3: CLIP EXTRACTION (GPU)
   └─> Wait for Stage 2 output
   └─> Load event CSV for date
   └─> For each event:
       ├─> Find source video
       ├─> Extract clip with ffmpeg
       ├─> Run YOLO on clip
       ├─> Add bounding boxes
       ├─> Compress with H.264
       └─> Save clip and detections CSV
   └─> Update state: Stage 3 complete

5. COMPLETION
   └─> All stages done for video
   └─> Log final metrics
   └─> Move to next video
```

### 4.2 File System Organization

```
Input Videos:
/mnt/BSP_NAS2_vol4/Video/Video{year}/{station}/{date}/
  ├── {station}_{timestamp}.mkv
  ├── {station}_{timestamp}.mp4
  └── ...

Stage 1 Output (Inference):
/mnt/BSP_NAS2_work/auklab_model/inference/{year}/{model_name}/{station}/
  ├── {station}_{timestamp}_raw.csv
  └── ...

Stage 2 Output (Events):
/mnt/BSP_NAS2_work/auklab_model/summarized_inference/{year}/{model_version}/{station}/{date}/
  ├── csv/
  │   ├── daily_events_{date}.csv
  │   ├── daily_flaps_{date}.csv
  │   ├── daily_per_second_{date}.csv
  │   └── daily_movement_{date}.csv
  ├── plots/
  │   └── daily_overview_{date}.png
  └── daily_summary_{date}.txt

Stage 3 Output (Clips):
/mnt/BSP_NAS2_work/auklab_model/event_data/{station}/{date}/
  ├── arrival_with_fish/
  │   ├── video/{event_id}.mp4
  │   └── csv/{event_id}_detections.csv
  ├── arrival_no_fish/
  │   ├── video/{event_id}.mp4
  │   └── csv/{event_id}_detections.csv
  └── departure/
      ├── video/{event_id}.mp4
      └── csv/{event_id}_detections.csv

System State:
/Users/jonas/Documents/Programming/python/Auklab_ObjectDetection/data/
  └── processing_state.db

Logs:
/Users/jonas/Documents/Programming/python/Auklab_ObjectDetection/logs/
  ├── master.log
  ├── scheduler.log
  ├── workers/...
  ├── errors.log
  ├── performance.log
  └── monitoring.log
```

---

## 5. ERROR HANDLING AND RECOVERY

### 5.1 Error Categories

#### **1. Recoverable Errors (Retry)**
- Network timeouts (NAS access)
- Temporary GPU memory issues
- Corrupted video file (skip frame)
- YOLO inference timeout

**Strategy**: Retry up to `max_retries` with exponential backoff

#### **2. Permanent Errors (Skip)**
- Video file not found
- Unsupported video codec
- Zero-byte video file
- Malformed timestamp in filename

**Strategy**: Mark as failed, log details, skip, continue with next video

#### **3. Critical Errors (Stop)**
- Configuration file corrupted
- State database corrupted
- Model file not found
- Out of disk space

**Strategy**: Log error, stop gracefully, notify user

### 5.2 Retry Mechanism

```python
def process_with_retry(job: VideoJob, processor: callable, max_retries: int) -> Result:
    """Generic retry wrapper"""
    for attempt in range(max_retries + 1):
        try:
            result = processor(job)
            return result
        except RecoverableError as e:
            if attempt < max_retries:
                delay = 2 ** attempt * 60  # Exponential backoff: 60s, 120s, 240s
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Max retries exceeded for {job.video_id}: {e}")
                raise PermanentError(f"Failed after {max_retries} retries") from e
        except PermanentError as e:
            logger.error(f"Permanent error for {job.video_id}: {e}")
            raise
```

### 5.3 Stuck Job Detection

On system startup:
1. Find all jobs marked as "in_progress" with timestamp > 1 hour ago
2. Assume worker crashed
3. Reset to "pending" status
4. Clear worker_id
5. Reset retry count if it was first attempt

### 5.4 Video Corruption Handling

- Use `try-except` around video file opening
- If PyAV fails to open, mark as skipped with error details
- If frames are successfully read but inference fails, retry
- Log corrupted video paths for manual inspection

---

## 6. OPTIMIZATION STRATEGIES

### 6.1 GPU Utilization

#### **Current Issue**: Single GPU per script invocation

**Optimizations**:
1. **Parallel GPU Workers**: 
   - Run 2 Stage 1 workers (one per GPU)
   - Run 2 Stage 3 workers (one per GPU)
   - Each worker processes different videos concurrently

2. **Model Loading**:
   - Load YOLO model once per worker (at startup)
   - Keep model in GPU memory throughout worker lifetime
   - Avoid repeated model loading overhead

3. **Batch Size Tuning**:
   - Current: 32 frames per batch
   - Test larger batches (64, 128) if GPU memory allows
   - Monitor GPU utilization with `nvidia-smi`

### 6.2 CPU Utilization

#### **Current Issue**: Event detection is single-threaded

**Optimizations**:
1. **Parallel CPU Workers**:
   - Run 8-16 Stage 2 workers (configurable)
   - Each processes different dates
   - CPU-bound operations (pandas aggregations, NumPy computations)

2. **Process Pool**:
   - Use `multiprocessing.Pool` with `cpu_count()` workers
   - Bypass Python GIL for true parallelism

### 6.3 I/O Optimization

#### **Disk I/O Bottleneck**: Reading videos from NAS

**Optimizations**:
1. **Read-ahead Caching**:
   - Prefetch next video while processing current
   - Use separate thread for I/O

2. **Sequential Access**:
   - Sort videos by station/date to minimize seek times
   - Process videos from same directory consecutively

3. **Output Buffering**:
   - Buffer CSV writes
   - Flush periodically or on completion

### 6.4 Pipeline Parallelism

**Key Insight**: Stages can run concurrently on different videos

```
Time  GPU0        GPU1        CPU0        CPU1
----  ----        ----        ----        ----
T0    Inf(V1)     Inf(V2)     -           -
T1    Inf(V3)     Inf(V4)     Evt(V1)     Evt(V2)
T2    Clip(V1)    Clip(V2)    Evt(V3)     Evt(V4)
T3    Clip(V5)    Clip(V6)    Evt(V5)     Evt(V6)
```

**Implementation**:
- Stage 2 workers poll State DB for completed Stage 1 jobs
- Stage 3 workers poll State DB for completed Stage 2 jobs
- Workers sleep briefly between polls (1-5 seconds)
- Use event-based notification (optional) for immediate triggering

---

## 7. TESTING STRATEGY

### 7.1 Unit Tests

**Modules to Test**:
- `config_manager.py`: Config loading, validation
- `state_manager.py`: Database operations, query logic
- `job_scheduler.py`: Priority calculation, video discovery
- Pipeline stages: Processing logic with mock inputs

**Test Framework**: `pytest`

**Example Test**:
```python
def test_priority_calculation():
    config = load_test_config()
    job = VideoJob(station="TRI3", year=2025, date="2025-06-30", ...)
    scheduler = JobScheduler(config, mock_state_mgr)
    priority = scheduler.calculate_priority(job)
    assert priority > 10000  # High priority station/year
```

### 7.2 Integration Tests

**Test Scenarios**:
1. **End-to-End Pipeline**: Process 5 test videos through all 3 stages
2. **Resume Logic**: Start processing, stop mid-way, resume, verify completion
3. **Error Handling**: Inject failures, verify retry and skip behavior
4. **Parallel Processing**: Run with 2 workers, verify no race conditions

**Test Data**: Create small test dataset (5-10 short videos)

### 7.3 Load Testing

**Objectives**:
- Measure throughput (videos/hour)
- Identify bottlenecks
- Validate GPU/CPU utilization

**Approach**:
- Process 100 representative videos
- Monitor with `nvidia-smi`, `htop`, `iotop`
- Adjust worker counts based on results

---

## 8. DEPLOYMENT AND OPERATION

### 8.1 Installation

**Prerequisites**:
- Python 3.8+
- CUDA 11.x+ (for GPU support)
- ffmpeg 4.x+
- System packages: `nvidia-smi` (GPU monitoring)

**Python Dependencies** (`requirements.txt`):
```
# Core
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.7.0
pandas>=1.5.0
numpy>=1.23.0
PyYAML>=6.0
pydantic>=2.0.0

# Video processing
av>=10.0.0

# Visualization
matplotlib>=3.6.0

# Optional
tqdm>=4.65.0
```

**Installation Steps**:
```bash
# 1. Clone repository
git clone <repo_url>
cd Auklab_ObjectDetection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directories
mkdir -p logs data config

# 5. Generate default config
python -m code.inference_system.config_manager --generate-default > config/system_config.yaml

# 6. Edit config with your paths
nano config/system_config.yaml
```

### 8.2 Running in Production

#### **Option 1: Terminal with `screen`** (Recommended for Remote)

```bash
# Start screen session
screen -S auklab_inference

# Activate environment
source venv/bin/activate

# Run system
python code/inference_system/main_orchestrator.py --config config/system_config.yaml

# Detach: Press Ctrl+A then D

# Reattach later
screen -r auklab_inference

# View logs in real-time
tail -f logs/monitoring.log
```

#### **Option 2: `tmux`** (Alternative to screen)

```bash
tmux new -s auklab_inference
python code/inference_system/main_orchestrator.py --config config/system_config.yaml
# Detach: Ctrl+B then D
tmux attach -t auklab_inference
```

#### **Option 3: `nohup`** (Background process)

```bash
nohup python code/inference_system/main_orchestrator.py \
    --config config/system_config.yaml \
    > output.log 2>&1 &

# Get PID
echo $! > auklab_inference.pid

# Stop later
kill $(cat auklab_inference.pid)
```

#### **Option 4: systemd Service** (Linux systems)

Create `/etc/systemd/system/auklab-inference.service`:
```ini
[Unit]
Description=Auklab Video Inference System
After=network.target

[Service]
Type=simple
User=jonas
WorkingDirectory=/Users/jonas/Documents/Programming/python/Auklab_ObjectDetection
ExecStart=/Users/jonas/Documents/Programming/python/Auklab_ObjectDetection/venv/bin/python code/inference_system/main_orchestrator.py --config config/system_config.yaml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable auklab-inference
sudo systemctl start auklab-inference
sudo systemctl status auklab-inference
```

### 8.3 Monitoring Operations

#### **Check Progress**
```bash
# View monitoring log
tail -f logs/monitoring.log

# Query state database
python code/inference_system/state_manager.py --status

# Output:
# Total Videos: 12,450
# Completed: 8,320 (66.8%)
# In Progress: 18
# Pending: 4,112 (33.2%)
# Failed: 0
# ETA: 6h 40m
```

#### **View Recent Errors**
```bash
tail -100 logs/errors.log
```

#### **Check Worker Status**
```bash
# List active worker logs
ls -lh logs/workers/

# View specific worker
tail -f logs/workers/gpu0-stage1.log
```

#### **Monitor GPU Usage**
```bash
watch -n 1 nvidia-smi
```

### 8.4 Stopping and Resuming

#### **Graceful Stop**
- Send SIGTERM or press Ctrl+C
- System will:
  1. Stop accepting new jobs
  2. Wait for current jobs to complete (up to 5 minutes)
  3. Save final state
  4. Cleanup resources
  5. Exit

#### **Force Stop**
- Send SIGKILL or press Ctrl+C twice
- Current jobs will be marked as stuck
- Will be retried on resume

#### **Resume**
```bash
python code/inference_system/main_orchestrator.py \
    --config config/system_config.yaml \
    --resume
```
- Loads state from database
- Resets stuck jobs
- Continues from where it left off

---

## 9. MAINTENANCE AND TROUBLESHOOTING

### 9.1 Common Issues

#### **Issue: "Model file not found"**
**Solution**: 
- Verify `detection_model` path in config
- Ensure model file exists: `ls -lh models/auklab_model_xlarge_combined_6080_v1.pt`

#### **Issue: "State database locked"**
**Solution**:
- Another instance is running: `ps aux | grep main_orchestrator`
- Kill other instance or wait for completion
- Remove lock file: `rm data/processing_state.db-lock`

#### **Issue: "GPU out of memory"**
**Solution**:
- Reduce `batch_size` in config (32 → 16 → 8)
- Restart system to clear GPU memory
- Check for memory leaks: `nvidia-smi` every minute

#### **Issue: "Videos not being discovered"**
**Solution**:
- Check video paths in config
- Verify NAS mount: `ls /mnt/BSP_NAS2_vol4/Video/`
- Check date range filters
- Enable debug logging: `--log-level DEBUG`

#### **Issue: "Processing very slow"**
**Solution**:
- Check GPU utilization: `nvidia-smi`
- Check disk I/O: `iotop`
- Increase worker counts if resources available
- Check NAS network speed: `iperf3`

### 9.2 Log Analysis

#### **Find Failed Videos**
```bash
grep "Failed" logs/errors.log | awk '{print $5}' | sort | uniq
```

#### **Performance Summary**
```bash
grep "Completed:" logs/workers/* | \
  awk -F'in ' '{print $2}' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count, "seconds"}'
```

#### **Error Rate by Hour**
```bash
awk -F'[][]' '/ERROR/ {print $2}' logs/errors.log | \
  cut -d' ' -f2 | cut -d':' -f1 | \
  sort | uniq -c
```

### 9.3 Database Maintenance

#### **Backup State Database**
```bash
# Daily backup
cp data/processing_state.db data/backups/processing_state_$(date +%Y%m%d).db
```

#### **Reset Specific Station**
```python
from code.inference_system.state_manager import StateManager

state_mgr = StateManager("data/processing_state.db")
state_mgr.reset_station("TRI3")  # Marks all TRI3 videos as pending
```

#### **Export State to CSV**
```python
state_mgr.export_status_csv("processing_status.csv")
```

### 9.4 Performance Tuning

After initial run, analyze performance logs:

1. **Identify slowest stage**: 
   - Check average processing time per stage
   - Allocate more workers to slowest stage

2. **Optimize batch size**:
   - Monitor GPU memory usage
   - Increase batch size until GPU saturated (80-90% utilization)

3. **Adjust worker counts**:
   - Stage 2 (CPU): Start with CPU count / 2
   - Monitor CPU usage, increase if < 70%

4. **Tune I/O**:
   - If disk I/O is bottleneck, reduce concurrent workers
   - Consider SSD cache for frequently accessed videos

---

## 10. FUTURE ENHANCEMENTS

### 10.1 Short-term (Phase 2)

1. **Web Dashboard**: Real-time monitoring UI with Flask/FastAPI
2. **Email Notifications**: Daily progress reports, error alerts
3. **Distributed Processing**: Support multiple compute nodes
4. **Cloud Storage**: S3/Azure Blob support for outputs
5. **Model Versioning**: Track which model version processed each video

### 10.2 Long-term (Phase 3)

1. **Active Learning Integration**: Flag uncertain detections for manual review
2. **Automated Quality Check**: Validate outputs, detect anomalies
3. **REST API**: Query status, trigger processing via API
4. **Docker Containerization**: Easy deployment
5. **Cloud Deployment**: AWS/GCP batch processing
6. **Advanced Scheduling**: Time-based (off-peak hours), cost-based (cloud)

---

## 11. IMPLEMENTATION TIMELINE

### Phase 1: Core Infrastructure (Week 1-2)
- Day 1-2: Configuration Manager + State Manager
- Day 3-4: Job Scheduler
- Day 5-7: Worker Pool Manager
- Day 8-10: Refactor Stage 1 (Inference)
- Day 11-12: Refactor Stage 2 (Events)
- Day 13-14: Refactor Stage 3 (Clips)

### Phase 2: Integration (Week 3)
- Day 15-16: Main Orchestrator
- Day 17-18: Logging System
- Day 19-20: Monitoring System
- Day 21: Integration testing

### Phase 3: Testing and Optimization (Week 4)
- Day 22-23: Unit tests
- Day 24-25: Integration tests with real data (100 videos)
- Day 26-27: Performance tuning
- Day 28: Documentation and deployment

### Phase 4: Production Deployment (Week 5)
- Day 29: Installation on production system
- Day 30: Test run with single station
- Day 31-35: Full production run with monitoring

---

## 12. SUCCESS METRICS

### 12.1 Performance Targets

- **Throughput**: Process 5000+ videos per day (with 2 GPUs)
- **Reliability**: < 1% failure rate (excluding corrupted videos)
- **Efficiency**: > 80% GPU utilization during Stage 1 and 3
- **Resumability**: Resume within 1 minute after interruption

### 12.2 Quality Targets

- **Stage 1**: Detection recall > 95% (validated on sample)
- **Stage 2**: Event detection precision > 90%
- **Stage 3**: Clips correctly extracted > 99%

### 12.3 Operational Targets

- **Uptime**: 24/7 unattended operation
- **Monitoring**: Status update every 60 seconds
- **Logging**: Comprehensive logs with < 1GB/day growth
- **Recovery**: Automatic recovery from transient failures

---

## 13. CONCLUSION

This implementation plan provides a comprehensive blueprint for building a robust, scalable, and maintainable video inference system. The modular architecture ensures:

- **Scalability**: Easy to add more GPUs/CPUs
- **Reliability**: Comprehensive error handling and state persistence
- **Maintainability**: Clean separation of concerns, well-documented
- **Operability**: Headless operation, resume support, monitoring

**Next Steps**:
1. Review this plan with stakeholders
2. Setup development environment
3. Begin Phase 1 implementation
4. Iterate based on testing results

**Key Success Factors**:
- Robust state management for resumability
- Efficient multi-GPU utilization
- Comprehensive logging for debugging
- Thorough testing before production deployment

---

## APPENDIX A: File Structure

```
Auklab_ObjectDetection/
├── code/
│   ├── inference_system/        # NEW: Integrated system
│   │   ├── __init__.py
│   │   ├── main_orchestrator.py
│   │   ├── config_manager.py
│   │   ├── state_manager.py
│   │   ├── job_scheduler.py
│   │   ├── worker_pool.py
│   │   ├── stage1_inference.py
│   │   ├── stage2_events.py
│   │   ├── stage3_clips.py
│   │   ├── logging_config.py
│   │   └── monitoring.py
│   │
│   ├── model/
│   │   └── run_inference_nth_decode.py  # KEEP: Reference
│   │
│   └── postprocess/
│       ├── batch_analyze_days.py  # KEEP: Reference
│       ├── extract_event_clips.py  # KEEP: Reference
│       └── event_detector.py
│
├── config/
│   ├── system_config.yaml          # Main config
│   └── system_config.template.yaml
│
├── data/
│   ├── processing_state.db         # State database
│   └── backups/
│
├── logs/
│   ├── master.log
│   ├── scheduler.log
│   ├── workers/
│   ├── errors.log
│   ├── performance.log
│   └── monitoring.log
│
├── models/
│   └── auklab_model_xlarge_combined_6080_v1.pt
│
├── tests/
│   ├── test_config_manager.py
│   ├── test_state_manager.py
│   ├── test_job_scheduler.py
│   └── test_integration.py
│
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md (this file)
```

---

## APPENDIX B: Dependencies

```
# requirements.txt
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.7.0
pandas>=1.5.0
numpy>=1.23.0
PyYAML>=6.0
pydantic>=2.0.0
av>=10.0.0
matplotlib>=3.6.0
tqdm>=4.65.0
psutil>=5.9.0  # For system monitoring
```

---

## APPENDIX C: Configuration Template

See Section 3.1 for full YAML configuration template.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-09  
**Author**: GitHub Copilot  
**Status**: Ready for Implementation
