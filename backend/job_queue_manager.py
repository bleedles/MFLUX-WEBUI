"""
MFLUX WebUI - Job Queue Manager
Manages the job queue with persistence, thread-safe operations, and status tracking.
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from backend.job_types import Job, JobStatus, JobType, JobPriority, get_job_type_display_name


class JobQueueManager:
    """
    Singleton manager for the job queue.
    Handles queue operations, persistence, and status tracking.
    """

    # Constants
    MAX_HISTORY_SIZE = 100
    DEFAULT_STEP_DURATION_SECONDS = {
        # Approximate seconds per step for different job types
        JobType.TEXT_TO_IMAGE_SIMPLE: 4.0,
        JobType.TEXT_TO_IMAGE_ADVANCED: 4.0,
        JobType.CONTROLNET: 5.0,
        JobType.IMAGE_TO_IMAGE: 4.0,
        JobType.IN_CONTEXT_LORA: 5.0,
        JobType.FILL: 6.0,
        JobType.DEPTH: 5.0,
        JobType.REDUX: 4.0,
        JobType.UPSCALE: 8.0,
        JobType.FLUX2_GENERATE: 3.0,
        JobType.FLUX2_EDIT: 4.0,
        JobType.QWEN_IMAGE: 5.0,
        JobType.QWEN_EDIT: 5.0,
        JobType.FIBO: 4.0,
        JobType.Z_IMAGE_TURBO: 2.0,
        JobType.KONTEXT: 5.0,
        JobType.IC_EDIT: 5.0,
        JobType.CATVTON: 6.0,
        JobType.CONCEPT_ATTENTION: 5.0,
    }

    def __init__(self, queue_dir: str = "configs"):
        """Initialize the job queue manager."""
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        self.queue_file = self.queue_dir / "job_queue.json"

        # Thread safety
        self._lock = threading.Lock()

        # Queue state
        self._pending_jobs: List[Job] = []
        self._active_job: Optional[Job] = None
        self._completed_jobs: List[Job] = []  # Recent history
        self._is_paused: bool = False

        # Callbacks for UI updates
        self._on_queue_changed: Optional[Callable[[], None]] = None

        # Load persisted queue
        self._load_queue()

    # ================== Queue Operations ==================

    def add_job(self, job: Job) -> str:
        """
        Add a new job to the queue.
        Returns the job ID.
        """
        with self._lock:
            # Estimate duration
            job.estimated_duration_seconds = self._estimate_job_duration(job)

            # Add to pending queue
            self._pending_jobs.append(job)

            # Sort by priority (higher priority first)
            self._sort_by_priority()

            # Persist
            self._save_queue()

            # Notify UI
            self._notify_queue_changed()

            return job.id

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        Returns True if job was found and cancelled.
        """
        with self._lock:
            # Check pending jobs
            for i, job in enumerate(self._pending_jobs):
                if job.id == job_id:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    self._pending_jobs.pop(i)
                    self._add_to_history(job)
                    self._save_queue()
                    self._notify_queue_changed()
                    return True

            # Check active job
            if self._active_job and self._active_job.id == job_id:
                self._active_job.status = JobStatus.CANCELLED
                # Note: The executor should check job status and stop execution
                self._save_queue()
                self._notify_queue_changed()
                return True

            return False

    def reorder_jobs(self, job_ids: List[str]) -> bool:
        """
        Reorder pending jobs according to the provided ID list.
        Jobs not in the list keep their relative order at the end.
        Returns True if successful.
        """
        with self._lock:
            if not job_ids:
                return False

            # Create a mapping of ID to job
            job_map = {job.id: job for job in self._pending_jobs}

            # Build new order
            new_order = []
            for job_id in job_ids:
                if job_id in job_map:
                    new_order.append(job_map[job_id])
                    del job_map[job_id]

            # Add remaining jobs (not in the provided list)
            new_order.extend(job_map.values())

            self._pending_jobs = new_order
            self._save_queue()
            self._notify_queue_changed()
            return True

    def move_job_up(self, job_id: str) -> bool:
        """Move a job one position up in the queue."""
        with self._lock:
            for i, job in enumerate(self._pending_jobs):
                if job.id == job_id and i > 0:
                    self._pending_jobs[i], self._pending_jobs[i - 1] = (
                        self._pending_jobs[i - 1],
                        self._pending_jobs[i],
                    )
                    self._save_queue()
                    self._notify_queue_changed()
                    return True
            return False

    def move_job_down(self, job_id: str) -> bool:
        """Move a job one position down in the queue."""
        with self._lock:
            for i, job in enumerate(self._pending_jobs):
                if job.id == job_id and i < len(self._pending_jobs) - 1:
                    self._pending_jobs[i], self._pending_jobs[i + 1] = (
                        self._pending_jobs[i + 1],
                        self._pending_jobs[i],
                    )
                    self._save_queue()
                    self._notify_queue_changed()
                    return True
            return False

    def get_next_job(self) -> Optional[Job]:
        """
        Get the next job to execute.
        Returns None if queue is empty or paused.
        """
        with self._lock:
            if self._is_paused or not self._pending_jobs:
                return None

            # Get the first pending job
            job = self._pending_jobs.pop(0)
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            # Set as active job
            self._active_job = job

            self._save_queue()
            self._notify_queue_changed()
            return job

    def clear_pending(self) -> int:
        """
        Clear all pending jobs.
        Returns the number of jobs cleared.
        """
        with self._lock:
            count = len(self._pending_jobs)

            # Move all to history as cancelled
            for job in self._pending_jobs:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self._add_to_history(job)

            self._pending_jobs = []
            self._save_queue()
            self._notify_queue_changed()
            return count

    def clear_history(self) -> int:
        """
        Clear job history.
        Returns the number of jobs cleared.
        """
        with self._lock:
            count = len(self._completed_jobs)
            self._completed_jobs = []
            self._save_queue()
            self._notify_queue_changed()
            return count

    # ================== Queue Control ==================

    def pause_queue(self) -> None:
        """Pause the queue (stops processing new jobs)."""
        with self._lock:
            self._is_paused = True
            self._save_queue()
            self._notify_queue_changed()

    def resume_queue(self) -> None:
        """Resume the queue."""
        with self._lock:
            self._is_paused = False
            self._save_queue()
            self._notify_queue_changed()

    def is_paused(self) -> bool:
        """Check if queue is paused."""
        with self._lock:
            return self._is_paused

    # ================== Status Methods ==================

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get complete queue status for UI display.
        """
        with self._lock:
            return {
                "is_paused": self._is_paused,
                "pending_count": len(self._pending_jobs),
                "active_job": self._active_job.to_dict() if self._active_job else None,
                "pending_jobs": [job.to_dict() for job in self._pending_jobs],
                "history": [job.to_dict() for job in self._completed_jobs[-20:]],  # Last 20
                "history_count": len(self._completed_jobs),
                "estimated_total_time": self._estimate_queue_time(),
            }

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID from any list."""
        with self._lock:
            # Check active
            if self._active_job and self._active_job.id == job_id:
                return self._active_job

            # Check pending
            for job in self._pending_jobs:
                if job.id == job_id:
                    return job

            # Check history
            for job in self._completed_jobs:
                if job.id == job_id:
                    return job

            return None

    def get_pending_jobs(self) -> List[Job]:
        """Get list of pending jobs."""
        with self._lock:
            return list(self._pending_jobs)

    def get_active_job(self) -> Optional[Job]:
        """Get the currently active job."""
        with self._lock:
            return self._active_job

    def get_history(self, limit: int = 50) -> List[Job]:
        """Get job history (most recent first)."""
        with self._lock:
            return list(reversed(self._completed_jobs[-limit:]))

    def get_pending_count(self) -> int:
        """Get number of pending jobs."""
        with self._lock:
            return len(self._pending_jobs)

    def has_active_job(self) -> bool:
        """Check if there's an active job."""
        with self._lock:
            return self._active_job is not None

    # ================== Progress Updates ==================

    def update_job_progress(
        self,
        job_id: str,
        progress: float,
        current_step: int = 0,
        total_steps: int = 0,
        message: str = "",
    ) -> None:
        """Update progress for a running job."""
        with self._lock:
            if self._active_job and self._active_job.id == job_id:
                self._active_job.progress = progress
                self._active_job.current_step = current_step
                self._active_job.total_steps = total_steps
                self._active_job.progress_message = message
                # Don't save on every progress update (too frequent)
                self._notify_queue_changed()

    def mark_job_completed(self, job_id: str, output_paths: List[str]) -> None:
        """Mark a job as completed with its output paths."""
        with self._lock:
            if self._active_job and self._active_job.id == job_id:
                self._active_job.status = JobStatus.COMPLETED
                self._active_job.completed_at = datetime.now()
                self._active_job.output_paths = output_paths
                self._active_job.progress = 1.0

                self._add_to_history(self._active_job)
                self._active_job = None

                self._save_queue()
                self._notify_queue_changed()

    def mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with error message."""
        with self._lock:
            if self._active_job and self._active_job.id == job_id:
                self._active_job.status = JobStatus.FAILED
                self._active_job.completed_at = datetime.now()
                self._active_job.error_message = error

                self._add_to_history(self._active_job)
                self._active_job = None

                self._save_queue()
                self._notify_queue_changed()

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled (for executor to check during execution)."""
        with self._lock:
            if self._active_job and self._active_job.id == job_id:
                return self._active_job.status == JobStatus.CANCELLED
            return True  # Job not found, consider it cancelled

    # ================== Time Estimation ==================

    def _estimate_job_duration(self, job: Job) -> float:
        """Estimate duration for a single job in seconds."""
        # Get base step duration for job type
        base_duration = self.DEFAULT_STEP_DURATION_SECONDS.get(job.job_type, 5.0)

        # Get steps from parameters
        steps = job.parameters.get("steps", 20)

        # Get number of images
        num_images = job.parameters.get("num_images", 1)

        # Base calculation
        duration = base_duration * steps * num_images

        # Adjust for image dimensions (larger = slower)
        width = job.parameters.get("width", 512)
        height = job.parameters.get("height", 512)
        dimension_factor = (width * height) / (512 * 512)
        duration *= max(0.5, min(2.0, dimension_factor ** 0.5))

        # Adjust for LoRAs (adds overhead)
        lora_files = job.parameters.get("lora_files", [])
        if lora_files:
            duration *= 1.0 + (len(lora_files) * 0.1)

        return duration

    def _estimate_queue_time(self) -> float:
        """Estimate total time to complete all pending jobs."""
        total = 0.0

        # Active job remaining time
        if self._active_job:
            if self._active_job.estimated_duration_seconds:
                remaining = self._active_job.estimated_duration_seconds * (
                    1.0 - self._active_job.progress
                )
                total += remaining

        # Pending jobs
        for job in self._pending_jobs:
            if job.estimated_duration_seconds:
                total += job.estimated_duration_seconds
            else:
                total += self._estimate_job_duration(job)

        return total

    # ================== Persistence ==================

    def _save_queue(self) -> None:
        """Save queue state to JSON file."""
        try:
            data = {
                "version": 1,
                "is_paused": self._is_paused,
                "active_job": self._active_job.to_dict() if self._active_job else None,
                "pending_jobs": [job.to_dict() for job in self._pending_jobs],
                "history": [job.to_dict() for job in self._completed_jobs],
            }

            with open(self.queue_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving job queue: {e}")

    def _load_queue(self) -> None:
        """Load queue state from JSON file."""
        try:
            if not self.queue_file.exists():
                return

            with open(self.queue_file, "r") as f:
                data = json.load(f)

            self._is_paused = data.get("is_paused", False)

            # Load active job (if it was running when app closed, mark as failed)
            active_data = data.get("active_job")
            if active_data:
                job = Job.from_dict(active_data)
                # Job was interrupted - mark as failed
                job.status = JobStatus.FAILED
                job.error_message = "Job interrupted (application was closed)"
                job.completed_at = datetime.now()
                self._completed_jobs.append(job)

            # Load pending jobs
            for job_data in data.get("pending_jobs", []):
                job = Job.from_dict(job_data)
                # Reset status to pending (in case it was marked otherwise)
                job.status = JobStatus.PENDING
                self._pending_jobs.append(job)

            # Load history
            for job_data in data.get("history", []):
                self._completed_jobs.append(Job.from_dict(job_data))

            # Trim history if needed
            self._trim_history()

        except Exception as e:
            print(f"Error loading job queue: {e}")

    # ================== Internal Helpers ==================

    def _sort_by_priority(self) -> None:
        """Sort pending jobs by priority (higher priority first)."""
        self._pending_jobs.sort(key=lambda j: j.priority.value, reverse=True)

    def _add_to_history(self, job: Job) -> None:
        """Add a job to history, trimming if needed."""
        self._completed_jobs.append(job)
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim history to MAX_HISTORY_SIZE."""
        if len(self._completed_jobs) > self.MAX_HISTORY_SIZE:
            self._completed_jobs = self._completed_jobs[-self.MAX_HISTORY_SIZE:]

    def _notify_queue_changed(self) -> None:
        """Notify listeners that queue has changed."""
        if self._on_queue_changed:
            try:
                self._on_queue_changed()
            except Exception as e:
                print(f"Error in queue change callback: {e}")

    def set_on_queue_changed(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for queue changes."""
        self._on_queue_changed = callback


# ================== Global Singleton ==================

_job_queue_manager: Optional[JobQueueManager] = None


def get_job_queue_manager() -> JobQueueManager:
    """Get the global JobQueueManager instance."""
    global _job_queue_manager
    if _job_queue_manager is None:
        _job_queue_manager = JobQueueManager()
    return _job_queue_manager


# ================== Convenience Functions ==================

def add_job_to_queue(
    job_type: JobType,
    parameters: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    name: str = "",
) -> str:
    """
    Convenience function to add a job to the queue.
    Returns the job ID.
    """
    manager = get_job_queue_manager()
    job = Job(
        job_type=job_type,
        parameters=parameters,
        priority=priority,
        name=name,
    )
    return manager.add_job(job)


def get_queue_status_for_ui() -> Dict[str, Any]:
    """Get queue status formatted for UI display."""
    manager = get_job_queue_manager()
    status = manager.get_queue_status()

    # Add formatted strings for UI
    status["estimated_time_formatted"] = _format_duration(status["estimated_total_time"])
    status["queue_badge"] = (
        f"{status['pending_count']}" if status["pending_count"] > 0 else ""
    )

    return status


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
