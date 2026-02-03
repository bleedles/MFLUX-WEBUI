"""
MFLUX WebUI - Queue Button Helper
Provides reusable "Add to Queue" button functionality for generation tabs.
"""

import gradio as gr
from typing import Dict, Any, Optional, Callable, List

from backend.job_queue_manager import add_job_to_queue
from backend.job_types import JobType, JobPriority


def create_add_to_queue_button(
    job_type: JobType,
    button_text: str = "📋 Add to Queue",
) -> gr.Button:
    """
    Create an "Add to Queue" button for a generation tab.

    Args:
        job_type: The JobType for this generation tab
        button_text: Text to display on the button

    Returns:
        A Gradio Button component
    """
    return gr.Button(
        button_text,
        variant="secondary",
        size="sm",
    )


def add_job_simple(
    prompt: str,
    model: str,
    image_format: str,
    lora_files: Optional[List[str]],
    ollama_model: Optional[str],
    system_prompt: Optional[str],
    num_images: int,
    low_ram: bool,
    *lora_scales,
) -> str:
    """Add a simple text-to-image job to the queue."""
    try:
        params = {
            "prompt": prompt,
            "model": model,
            "image_format": image_format,
            "lora_files": lora_files or [],
            "ollama_model": ollama_model,
            "system_prompt": system_prompt,
            "num_images": int(num_images) if num_images else 1,
            "low_ram": bool(low_ram),
            "lora_scales": list(lora_scales) if lora_scales else [],
        }

        job_id = add_job_to_queue(
            job_type=JobType.TEXT_TO_IMAGE_SIMPLE,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


def add_job_advanced(
    prompt: str,
    model: str,
    base_model: Optional[str],
    seed: Optional[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    lora_files: Optional[List[str]],
    metadata: bool,
    ollama_model: Optional[str],
    system_prompt: Optional[str],
    prompt_file: Optional[str],
    config_from_metadata: Optional[str],
    stepwise_output_dir: Optional[str],
    vae_tiling: bool,
    vae_tiling_split: int,
    num_images: int,
    low_ram: bool,
    auto_seeds: Optional[str],
    *lora_scales,
) -> str:
    """Add an advanced text-to-image job to the queue."""
    try:
        params = {
            "prompt": prompt,
            "model": model,
            "base_model": base_model,
            "seed": int(seed) if seed else None,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance) if guidance else 3.5,
            "lora_files": lora_files or [],
            "metadata": bool(metadata),
            "ollama_model": ollama_model,
            "system_prompt": system_prompt,
            "prompt_file": prompt_file,
            "config_from_metadata": config_from_metadata,
            "stepwise_output_dir": stepwise_output_dir,
            "vae_tiling": bool(vae_tiling),
            "vae_tiling_split": int(vae_tiling_split) if vae_tiling_split else 1,
            "num_images": int(num_images) if num_images else 1,
            "low_ram": bool(low_ram),
            "auto_seeds": auto_seeds,
            "lora_scales": list(lora_scales) if lora_scales else [],
        }

        job_id = add_job_to_queue(
            job_type=JobType.TEXT_TO_IMAGE_ADVANCED,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


def add_job_controlnet(
    prompt: str,
    control_image,
    model: str,
    controlnet_model: str,
    seed: Optional[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    controlnet_strength: float,
    lora_files: Optional[List[str]],
    metadata: bool,
    num_images: int,
    low_ram: bool,
    *lora_scales,
) -> str:
    """Add a ControlNet job to the queue."""
    try:
        # Handle control image - if it's a PIL Image, save it temporarily
        control_image_path = None
        if control_image is not None:
            if hasattr(control_image, 'save'):
                import tempfile
                import os
                temp_dir = "queue/temp_images"
                os.makedirs(temp_dir, exist_ok=True)
                control_image_path = f"{temp_dir}/control_{id(control_image)}.png"
                control_image.save(control_image_path)
            else:
                control_image_path = str(control_image)

        params = {
            "prompt": prompt,
            "control_image": control_image_path,
            "model": model,
            "controlnet_model": controlnet_model,
            "seed": int(seed) if seed else None,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance) if guidance else 3.5,
            "controlnet_strength": float(controlnet_strength) if controlnet_strength else 0.5,
            "lora_files": lora_files or [],
            "metadata": bool(metadata),
            "num_images": int(num_images) if num_images else 1,
            "low_ram": bool(low_ram),
            "lora_scales": list(lora_scales) if lora_scales else [],
        }

        job_id = add_job_to_queue(
            job_type=JobType.CONTROLNET,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


def add_job_image_to_image(
    prompt: str,
    init_image,
    model: str,
    seed: Optional[int],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    strength: float,
    lora_files: Optional[List[str]],
    metadata: bool,
    num_images: int,
    low_ram: bool,
    *lora_scales,
) -> str:
    """Add an image-to-image job to the queue."""
    try:
        # Handle init image
        init_image_path = None
        if init_image is not None:
            if hasattr(init_image, 'save'):
                import os
                temp_dir = "queue/temp_images"
                os.makedirs(temp_dir, exist_ok=True)
                init_image_path = f"{temp_dir}/init_{id(init_image)}.png"
                init_image.save(init_image_path)
            else:
                init_image_path = str(init_image)

        params = {
            "prompt": prompt,
            "init_image": init_image_path,
            "model": model,
            "seed": int(seed) if seed else None,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "guidance": float(guidance) if guidance else 3.5,
            "strength": float(strength) if strength else 0.75,
            "lora_files": lora_files or [],
            "metadata": bool(metadata),
            "num_images": int(num_images) if num_images else 1,
            "low_ram": bool(low_ram),
            "lora_scales": list(lora_scales) if lora_scales else [],
        }

        job_id = add_job_to_queue(
            job_type=JobType.IMAGE_TO_IMAGE,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


def add_job_flux2_generate(
    prompt: str,
    model_name: str,
    seed: Optional[int],
    width: int,
    height: int,
    steps: int,
    lora_files: Optional[List[str]],
    lora_scales: Optional[List[float]],
    metadata: bool,
    num_images: int,
) -> str:
    """Add a Flux2 generation job to the queue."""
    try:
        params = {
            "prompt": prompt,
            "model_name": model_name,
            "seed": int(seed) if seed else None,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "lora_files": lora_files or [],
            "lora_scales": lora_scales or [],
            "metadata": bool(metadata),
            "num_images": int(num_images) if num_images else 1,
        }

        job_id = add_job_to_queue(
            job_type=JobType.FLUX2_GENERATE,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


def add_job_generic(
    job_type: JobType,
    **kwargs,
) -> str:
    """
    Generic function to add any job type to the queue.
    Pass parameters as keyword arguments.
    """
    try:
        # Filter out None values and convert types as needed
        params = {}
        for key, value in kwargs.items():
            if value is not None:
                # Handle PIL Images
                if hasattr(value, 'save'):
                    import os
                    temp_dir = "queue/temp_images"
                    os.makedirs(temp_dir, exist_ok=True)
                    image_path = f"{temp_dir}/{key}_{id(value)}.png"
                    value.save(image_path)
                    params[key] = image_path
                else:
                    params[key] = value

        job_id = add_job_to_queue(
            job_type=job_type,
            parameters=params,
            priority=JobPriority.NORMAL,
        )

        return f"✅ Added to queue (Job ID: {job_id})"
    except Exception as e:
        return f"❌ Failed to add to queue: {str(e)}"


# Convenience function for creating a queue status display
def create_queue_status_display() -> gr.Textbox:
    """Create a textbox to display queue status after adding a job."""
    return gr.Textbox(
        label="Queue Status",
        value="",
        interactive=False,
        visible=True,
        lines=1,
    )
