"""
MFLUX WebUI - Job Types
Data classes and enums for the job queue system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


class JobStatus(Enum):
    """Status of a job in the queue."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of generation jobs supported by the queue."""
    # Core generation
    TEXT_TO_IMAGE_SIMPLE = "text_to_image_simple"
    TEXT_TO_IMAGE_ADVANCED = "text_to_image_advanced"
    CONTROLNET = "controlnet"
    IMAGE_TO_IMAGE = "image_to_image"
    IN_CONTEXT_LORA = "in_context_lora"

    # Fill/Inpaint
    FILL = "fill"

    # Depth
    DEPTH = "depth"

    # Redux (variations)
    REDUX = "redux"

    # Upscale
    UPSCALE = "upscale"

    # Flux2 models
    FLUX2_GENERATE = "flux2_generate"
    FLUX2_EDIT = "flux2_edit"

    # Qwen models
    QWEN_IMAGE = "qwen_image"
    QWEN_EDIT = "qwen_edit"

    # Other models
    FIBO = "fibo"
    Z_IMAGE_TURBO = "z_image_turbo"
    KONTEXT = "kontext"
    IC_EDIT = "ic_edit"
    CATVTON = "catvton"
    CONCEPT_ATTENTION = "concept_attention"


class JobPriority(Enum):
    """Priority levels for jobs. Higher value = higher priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Job:
    """Represents a generation job in the queue."""

    # Unique identifier (8-char UUID)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Job classification
    job_type: JobType = JobType.TEXT_TO_IMAGE_SIMPLE
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL

    # Generation parameters (stored as dict for flexibility across job types)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # User-friendly name (auto-generated from prompt if not provided)
    name: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    current_step: int = 0
    total_steps: int = 0
    progress_message: str = ""

    # Results
    output_paths: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    # Time estimation (in seconds)
    estimated_duration_seconds: Optional[float] = None

    def __post_init__(self):
        """Auto-generate name from prompt if not provided."""
        if not self.name and self.parameters.get("prompt"):
            prompt = self.parameters["prompt"]
            # Truncate to 50 chars and add ellipsis
            self.name = prompt[:50] + ("..." if len(prompt) > 50 else "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job to dictionary for JSON persistence."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "parameters": self._serialize_parameters(),
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_message": self.progress_message,
            "output_paths": self.output_paths,
            "error_message": self.error_message,
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }

    def _serialize_parameters(self) -> Dict[str, Any]:
        """Serialize parameters, handling special types like PIL Images."""
        serialized = {}
        for key, value in self.parameters.items():
            # Skip None values
            if value is None:
                continue
            # Handle PIL Images by storing paths (images should already be saved)
            if hasattr(value, 'save'):  # Duck-type check for PIL Image
                # Images should be pre-saved; if not, skip
                continue
            serialized[key] = value
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Deserialize job from dictionary."""
        return cls(
            id=data["id"],
            job_type=JobType(data["job_type"]),
            status=JobStatus(data["status"]),
            priority=JobPriority(data["priority"]),
            parameters=data.get("parameters", {}),
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=data.get("progress", 0.0),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            progress_message=data.get("progress_message", ""),
            output_paths=data.get("output_paths", []),
            error_message=data.get("error_message"),
            estimated_duration_seconds=data.get("estimated_duration_seconds"),
        )

    def get_duration_seconds(self) -> Optional[float]:
        """Get actual duration if job is completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_wait_time_seconds(self) -> float:
        """Get time spent waiting in queue."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()


# Job type to display name mapping
JOB_TYPE_DISPLAY_NAMES: Dict[JobType, str] = {
    JobType.TEXT_TO_IMAGE_SIMPLE: "Text to Image (Simple)",
    JobType.TEXT_TO_IMAGE_ADVANCED: "Text to Image (Advanced)",
    JobType.CONTROLNET: "ControlNet",
    JobType.IMAGE_TO_IMAGE: "Image to Image",
    JobType.IN_CONTEXT_LORA: "In-Context LoRA",
    JobType.FILL: "Fill/Inpaint",
    JobType.DEPTH: "Depth",
    JobType.REDUX: "Redux (Variations)",
    JobType.UPSCALE: "Upscale",
    JobType.FLUX2_GENERATE: "Flux2 Generate",
    JobType.FLUX2_EDIT: "Flux2 Edit",
    JobType.QWEN_IMAGE: "Qwen Image",
    JobType.QWEN_EDIT: "Qwen Edit",
    JobType.FIBO: "Fibo",
    JobType.Z_IMAGE_TURBO: "Z-Image Turbo",
    JobType.KONTEXT: "Kontext",
    JobType.IC_EDIT: "IC-Edit",
    JobType.CATVTON: "CatVTON",
    JobType.CONCEPT_ATTENTION: "Concept Attention",
}


def get_job_type_display_name(job_type: JobType) -> str:
    """Get human-readable display name for a job type."""
    return JOB_TYPE_DISPLAY_NAMES.get(job_type, job_type.value)
