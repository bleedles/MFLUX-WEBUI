"""
MFLUX WebUI - Job Executor
Background worker thread that executes jobs from the queue sequentially.
"""

import gc
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from PIL import Image

from backend.job_types import Job, JobType, JobStatus
from backend.job_queue_manager import get_job_queue_manager, JobQueueManager


class JobExecutor:
    """
    Background worker that processes jobs from the queue.
    Runs in a separate thread and executes jobs sequentially.
    """

    def __init__(self, queue_manager: Optional[JobQueueManager] = None):
        """Initialize the job executor."""
        self.queue_manager = queue_manager or get_job_queue_manager()

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Polling interval (seconds)
        self._poll_interval = 0.5

        # Generation function registry
        self._generation_functions: Dict[JobType, Callable] = {}
        self._register_generation_functions()

    def _register_generation_functions(self) -> None:
        """Register generation functions for each job type."""
        # Import generation functions lazily to avoid circular imports
        try:
            from backend.flux_manager import (
                simple_generate_image,
                generate_image_gradio,
                generate_image_controlnet_gradio,
                generate_image_i2i_gradio,
                generate_image_in_context_lora_gradio,
            )
            self._generation_functions[JobType.TEXT_TO_IMAGE_SIMPLE] = self._wrap_simple_generate
            self._generation_functions[JobType.TEXT_TO_IMAGE_ADVANCED] = self._wrap_advanced_generate
            self._generation_functions[JobType.CONTROLNET] = self._wrap_controlnet_generate
            self._generation_functions[JobType.IMAGE_TO_IMAGE] = self._wrap_i2i_generate
            self._generation_functions[JobType.IN_CONTEXT_LORA] = self._wrap_in_context_lora_generate
        except ImportError as e:
            print(f"Warning: Could not import flux_manager functions: {e}")

        try:
            from backend.flux2_manager import (
                generate_flux2_image_gradio,
                generate_flux2_edit_gradio,
            )
            self._generation_functions[JobType.FLUX2_GENERATE] = self._wrap_flux2_generate
            self._generation_functions[JobType.FLUX2_EDIT] = self._wrap_flux2_edit
        except ImportError as e:
            print(f"Warning: Could not import flux2_manager functions: {e}")

        try:
            from backend.fill_manager import generate_fill_gradio
            self._generation_functions[JobType.FILL] = self._wrap_fill_generate
        except ImportError as e:
            print(f"Warning: Could not import fill_manager: {e}")

        try:
            from backend.depth_manager import generate_depth_gradio
            self._generation_functions[JobType.DEPTH] = self._wrap_depth_generate
        except ImportError as e:
            print(f"Warning: Could not import depth_manager: {e}")

        try:
            from backend.redux_manager import generate_redux_gradio
            self._generation_functions[JobType.REDUX] = self._wrap_redux_generate
        except ImportError as e:
            print(f"Warning: Could not import redux_manager: {e}")

        try:
            from backend.upscale_manager import upscale_image_gradio
            self._generation_functions[JobType.UPSCALE] = self._wrap_upscale_generate
        except ImportError as e:
            print(f"Warning: Could not import upscale_manager: {e}")

        try:
            from backend.qwen_manager import (
                generate_qwen_image_gradio,
                generate_qwen_edit_gradio,
            )
            self._generation_functions[JobType.QWEN_IMAGE] = self._wrap_qwen_image_generate
            self._generation_functions[JobType.QWEN_EDIT] = self._wrap_qwen_edit_generate
        except ImportError as e:
            print(f"Warning: Could not import qwen_manager: {e}")

        try:
            from backend.fibo_manager import generate_fibo_image_gradio
            self._generation_functions[JobType.FIBO] = self._wrap_fibo_generate
        except ImportError as e:
            print(f"Warning: Could not import fibo_manager: {e}")

        try:
            from backend.z_image_manager import generate_z_image_gradio
            self._generation_functions[JobType.Z_IMAGE_TURBO] = self._wrap_z_image_generate
        except ImportError as e:
            print(f"Warning: Could not import z_image_manager: {e}")

        try:
            from backend.kontext_manager import generate_kontext_image_gradio
            self._generation_functions[JobType.KONTEXT] = self._wrap_kontext_generate
        except ImportError as e:
            print(f"Warning: Could not import kontext_manager: {e}")

        try:
            from backend.ic_edit_manager import generate_ic_edit_gradio
            self._generation_functions[JobType.IC_EDIT] = self._wrap_ic_edit_generate
        except ImportError as e:
            print(f"Warning: Could not import ic_edit_manager: {e}")

        try:
            from backend.catvton_manager import generate_catvton_gradio
            self._generation_functions[JobType.CATVTON] = self._wrap_catvton_generate
        except ImportError as e:
            print(f"Warning: Could not import catvton_manager: {e}")

        try:
            from backend.concept_attention_manager import generate_concept_attention_gradio
            self._generation_functions[JobType.CONCEPT_ATTENTION] = self._wrap_concept_attention_generate
        except ImportError as e:
            print(f"Warning: Could not import concept_attention_manager: {e}")

    # ================== Thread Control ==================

    def start(self) -> None:
        """Start the executor background thread."""
        if self._is_running:
            print("Job executor is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        print("Job executor started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the executor gracefully."""
        if not self._is_running:
            return

        print("Stopping job executor...")
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print("Warning: Job executor thread did not stop cleanly")

        self._is_running = False
        print("Job executor stopped")

    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._is_running

    # ================== Main Execution Loop ==================

    def _run_loop(self) -> None:
        """Main execution loop - runs in background thread."""
        print("Job executor loop started")

        while not self._stop_event.is_set():
            try:
                # Check if queue is paused
                if self.queue_manager.is_paused():
                    time.sleep(self._poll_interval)
                    continue

                # Get next job
                job = self.queue_manager.get_next_job()

                if job:
                    self._execute_job(job)
                    # Cleanup after job
                    self._cleanup_after_job()
                else:
                    # No job available, wait
                    time.sleep(self._poll_interval)

            except Exception as e:
                print(f"Error in job executor loop: {e}")
                traceback.print_exc()
                time.sleep(1.0)  # Wait before retrying

        print("Job executor loop exited")

    def _execute_job(self, job: Job) -> None:
        """Execute a single job."""
        print(f"\n{'='*50}")
        print(f"Executing job: {job.id} ({job.job_type.value})")
        print(f"Parameters: {job.parameters}")
        print(f"{'='*50}\n")

        try:
            # Check if job type is supported
            if job.job_type not in self._generation_functions:
                raise ValueError(f"Unsupported job type: {job.job_type.value}")

            # Get the generation function
            gen_func = self._generation_functions[job.job_type]

            # Execute the generation
            result = gen_func(job)

            # Handle result
            if result:
                images, output_paths = result
                self.queue_manager.mark_job_completed(job.id, output_paths)
                print(f"Job {job.id} completed successfully. Outputs: {output_paths}")
            else:
                self.queue_manager.mark_job_failed(job.id, "No result returned")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Job {job.id} failed: {error_msg}")
            traceback.print_exc()
            self.queue_manager.mark_job_failed(job.id, error_msg)

    def _cleanup_after_job(self) -> None:
        """Clean up resources after job completion."""
        try:
            # Force garbage collection
            gc.collect()

            # Try to clean up MLX memory if available
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass

        except Exception as e:
            print(f"Warning: Cleanup after job failed: {e}")

    # ================== Progress Callback Helper ==================

    def _make_progress_callback(self, job: Job) -> Callable[[float, int, int, str], None]:
        """Create a progress callback for a job."""
        def callback(progress: float, current_step: int = 0, total_steps: int = 0, message: str = ""):
            # Check if job was cancelled
            if self.queue_manager.is_job_cancelled(job.id):
                raise InterruptedError("Job was cancelled")

            self.queue_manager.update_job_progress(
                job.id, progress, current_step, total_steps, message
            )
        return callback

    # ================== Generation Wrappers ==================
    # Each wrapper extracts parameters and calls the appropriate generation function

    def _wrap_simple_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute simple text-to-image generation."""
        from backend.flux_manager import simple_generate_image

        params = job.parameters
        result = simple_generate_image(
            prompt=params.get("prompt", ""),
            model=params.get("model", "schnell"),
            image_format=params.get("image_format", "Square (512x512)"),
            lora_files=params.get("lora_files"),
            ollama_model=params.get("ollama_model"),
            system_prompt=params.get("system_prompt"),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            # Parse output paths from result string
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_advanced_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute advanced text-to-image generation."""
        from backend.flux_manager import generate_image_gradio

        params = job.parameters
        result = generate_image_gradio(
            prompt=params.get("prompt", ""),
            model=params.get("model", "schnell"),
            base_model=params.get("base_model"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            lora_files=params.get("lora_files"),
            metadata=params.get("metadata", True),
            ollama_model=params.get("ollama_model"),
            system_prompt=params.get("system_prompt"),
            prompt_file=params.get("prompt_file"),
            config_from_metadata=params.get("config_from_metadata"),
            stepwise_output_dir=params.get("stepwise_output_dir"),
            vae_tiling=params.get("vae_tiling", False),
            vae_tiling_split=params.get("vae_tiling_split", 1),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
            auto_seeds=params.get("auto_seeds"),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_controlnet_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute ControlNet generation."""
        from backend.flux_manager import generate_image_controlnet_gradio

        params = job.parameters
        result = generate_image_controlnet_gradio(
            prompt=params.get("prompt", ""),
            control_image=params.get("control_image"),
            model=params.get("model", "schnell"),
            controlnet_model=params.get("controlnet_model", "canny"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            controlnet_strength=params.get("controlnet_strength", 0.5),
            lora_files=params.get("lora_files"),
            metadata=params.get("metadata", True),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_i2i_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute image-to-image generation."""
        from backend.flux_manager import generate_image_i2i_gradio

        params = job.parameters
        result = generate_image_i2i_gradio(
            prompt=params.get("prompt", ""),
            init_image=params.get("init_image"),
            model=params.get("model", "schnell"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            strength=params.get("strength", 0.75),
            lora_files=params.get("lora_files"),
            metadata=params.get("metadata", True),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_in_context_lora_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute in-context LoRA generation."""
        from backend.flux_manager import generate_image_in_context_lora_gradio

        params = job.parameters
        result = generate_image_in_context_lora_gradio(
            prompt=params.get("prompt", ""),
            reference_image=params.get("reference_image"),
            model=params.get("model", "schnell"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            lora_files=params.get("lora_files"),
            metadata=params.get("metadata", True),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_flux2_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Flux2 text-to-image generation."""
        from backend.flux2_manager import generate_flux2_image_gradio

        params = job.parameters
        result = generate_flux2_image_gradio(
            prompt=params.get("prompt", ""),
            model_name=params.get("model_name", "flux2-klein-8bit"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            lora_files=params.get("lora_files"),
            lora_scales=params.get("lora_scales"),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_flux2_edit(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Flux2 edit generation."""
        from backend.flux2_manager import generate_flux2_edit_gradio

        params = job.parameters
        result = generate_flux2_edit_gradio(
            prompt=params.get("prompt", ""),
            init_image=params.get("init_image"),
            model_name=params.get("model_name", "flux2-klein-8bit"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            strength=params.get("strength", 0.75),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_fill_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute fill/inpaint generation."""
        from backend.fill_manager import generate_fill_gradio

        params = job.parameters
        result = generate_fill_gradio(
            prompt=params.get("prompt", ""),
            image_input=params.get("image_input"),
            mask_input=params.get("mask_input"),
            base_model=params.get("base_model"),
            seed=params.get("seed"),
            height=params.get("height", 512),
            width=params.get("width", 512),
            steps=params.get("steps", 50),
            guidance=params.get("guidance", 30.0),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_depth_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute depth-conditioned generation."""
        from backend.depth_manager import generate_depth_gradio

        params = job.parameters
        result = generate_depth_gradio(
            prompt=params.get("prompt", ""),
            depth_image=params.get("depth_image"),
            model=params.get("model", "schnell"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            depth_strength=params.get("depth_strength", 0.5),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_redux_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Redux (variation) generation."""
        from backend.redux_manager import generate_redux_gradio

        params = job.parameters
        result = generate_redux_gradio(
            prompt=params.get("prompt", ""),
            reference_image=params.get("reference_image"),
            model=params.get("model", "schnell"),
            seed=params.get("seed"),
            height=params.get("height", 512),
            width=params.get("width", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            redux_strength=params.get("redux_strength", 1.0),
            lora_files=params.get("lora_files"),
            metadata=params.get("metadata", True),
            *params.get("lora_scales", []),
            num_images=params.get("num_images", 1),
            low_ram=params.get("low_ram", False),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_upscale_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute upscale generation."""
        from backend.upscale_manager import upscale_image_gradio

        params = job.parameters
        result = upscale_image_gradio(
            input_image=params.get("input_image"),
            upscale_factor=params.get("upscale_factor", 2),
            model=params.get("model"),
            metadata=params.get("metadata", True),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_qwen_image_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Qwen image generation."""
        from backend.qwen_manager import generate_qwen_image_gradio

        params = job.parameters
        result = generate_qwen_image_gradio(
            prompt=params.get("prompt", ""),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 50),
            guidance=params.get("guidance", 7.5),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_qwen_edit_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Qwen edit generation."""
        from backend.qwen_manager import generate_qwen_edit_gradio

        params = job.parameters
        result = generate_qwen_edit_gradio(
            prompt=params.get("prompt", ""),
            init_image=params.get("init_image"),
            seed=params.get("seed"),
            steps=params.get("steps", 50),
            guidance=params.get("guidance", 7.5),
            strength=params.get("strength", 0.75),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_fibo_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Fibo generation."""
        from backend.fibo_manager import generate_fibo_image_gradio

        params = job.parameters
        result = generate_fibo_image_gradio(
            prompt=params.get("prompt", ""),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_z_image_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Z-Image Turbo generation."""
        from backend.z_image_manager import generate_z_image_gradio

        params = job.parameters
        result = generate_z_image_gradio(
            prompt=params.get("prompt", ""),
            negative_prompt=params.get("negative_prompt", ""),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 0.0),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_kontext_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Kontext generation."""
        from backend.kontext_manager import generate_kontext_image_gradio

        params = job.parameters
        result = generate_kontext_image_gradio(
            prompt=params.get("prompt", ""),
            reference_image=params.get("reference_image"),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_ic_edit_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute IC-Edit generation."""
        from backend.ic_edit_manager import generate_ic_edit_gradio

        params = job.parameters
        result = generate_ic_edit_gradio(
            prompt=params.get("prompt", ""),
            init_image=params.get("init_image"),
            seed=params.get("seed"),
            steps=params.get("steps", 50),
            guidance=params.get("guidance", 7.5),
            strength=params.get("strength", 0.75),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_catvton_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute CatVTON generation."""
        from backend.catvton_manager import generate_catvton_gradio

        params = job.parameters
        result = generate_catvton_gradio(
            person_image=params.get("person_image"),
            garment_image=params.get("garment_image"),
            seed=params.get("seed"),
            steps=params.get("steps", 50),
            guidance=params.get("guidance", 2.5),
            metadata=params.get("metadata", True),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    def _wrap_concept_attention_generate(self, job: Job) -> Optional[Tuple[List[Image.Image], List[str]]]:
        """Execute Concept Attention generation."""
        from backend.concept_attention_manager import generate_concept_attention_gradio

        params = job.parameters
        result = generate_concept_attention_gradio(
            prompt=params.get("prompt", ""),
            concepts=params.get("concepts", []),
            seed=params.get("seed"),
            width=params.get("width", 512),
            height=params.get("height", 512),
            steps=params.get("steps", 4),
            guidance=params.get("guidance", 3.5),
            metadata=params.get("metadata", True),
            num_images=params.get("num_images", 1),
        )

        if result and len(result) >= 2:
            images = result[0] if isinstance(result[0], list) else [result[0]]
            output_paths = self._parse_output_paths(result[1])
            return images, output_paths

        return None

    # ================== Helper Methods ==================

    def _parse_output_paths(self, result_string: str) -> List[str]:
        """Parse output file paths from result string."""
        if not result_string:
            return []

        # Result might be a newline-separated list of paths
        paths = []
        for line in str(result_string).strip().split("\n"):
            line = line.strip()
            if line and (line.endswith(".png") or line.endswith(".jpg") or line.endswith(".jpeg")):
                # Check if it's a full path or just filename
                if "/" in line or "\\" in line:
                    paths.append(line)
                else:
                    # Assume it's in the output directory
                    paths.append(f"output/{line}")

        return paths


# ================== Global Singleton ==================

_job_executor: Optional[JobExecutor] = None


def get_job_executor() -> JobExecutor:
    """Get the global JobExecutor instance."""
    global _job_executor
    if _job_executor is None:
        _job_executor = JobExecutor()
    return _job_executor


def start_job_executor() -> None:
    """Start the global job executor."""
    executor = get_job_executor()
    executor.start()


def stop_job_executor() -> None:
    """Stop the global job executor."""
    global _job_executor
    if _job_executor:
        _job_executor.stop()
