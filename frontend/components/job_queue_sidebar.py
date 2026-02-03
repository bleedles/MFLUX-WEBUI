"""
MFLUX WebUI - Job Queue Sidebar
Global sidebar UI component for managing the job queue.
Displays active job, pending jobs, and history with controls.
"""

import gradio as gr
from typing import Dict, Any, List, Tuple, Optional
import json

from backend.job_queue_manager import (
    get_job_queue_manager,
    get_queue_status_for_ui,
    JobQueueManager,
)
from backend.job_types import Job, JobStatus, JobPriority, get_job_type_display_name


def create_job_queue_sidebar():
    """
    Create the job queue sidebar component.
    Returns a dictionary of components for event wiring.
    """
    manager = get_job_queue_manager()

    with gr.Column(scale=1, min_width=280) as sidebar_column:
        # Collapsible accordion for the queue
        with gr.Accordion("📋 Job Queue", open=False) as queue_accordion:
            # Queue status summary
            queue_status_md = gr.Markdown(
                "**Queue Status**: Idle\n\nNo jobs in queue.",
                elem_id="queue-status-summary"
            )

            # Pause/Resume toggle
            with gr.Row():
                pause_btn = gr.Button("⏸️ Pause", size="sm", scale=1)
                resume_btn = gr.Button("▶️ Resume", size="sm", scale=1, visible=False)
                clear_pending_btn = gr.Button("🗑️ Clear", size="sm", variant="stop", scale=1)

            # Active job section
            gr.Markdown("### Active Job", elem_id="active-job-header")
            active_job_display = gr.Textbox(
                label="Currently Running",
                value="No active job",
                lines=2,
                interactive=False,
                show_label=False,
            )
            active_job_progress = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                label="Progress",
                interactive=False,
            )
            cancel_active_btn = gr.Button(
                "❌ Cancel Active Job",
                size="sm",
                variant="stop",
                visible=False,
            )

            # Pending jobs section
            gr.Markdown("### Pending Jobs", elem_id="pending-jobs-header")
            pending_jobs_display = gr.Textbox(
                label="Queued",
                value="No pending jobs",
                lines=4,
                interactive=False,
                show_label=False,
            )

            # Job selection for actions
            with gr.Row():
                selected_job_id = gr.Textbox(
                    label="Selected Job ID",
                    placeholder="Enter job ID",
                    scale=2,
                    visible=False,
                )
                move_up_btn = gr.Button("⬆️", size="sm", scale=1, visible=False)
                move_down_btn = gr.Button("⬇️", size="sm", scale=1, visible=False)
                cancel_job_btn = gr.Button("❌", size="sm", variant="stop", scale=1, visible=False)

            # History section (collapsed by default)
            with gr.Accordion("📜 History", open=False) as history_accordion:
                history_display = gr.Textbox(
                    label="Recent Jobs",
                    value="No completed jobs",
                    lines=6,
                    interactive=False,
                    show_label=False,
                )
                clear_history_btn = gr.Button("Clear History", size="sm")

            # Hidden state for tracking
            queue_state = gr.State({
                "active_job_id": None,
                "pending_job_ids": [],
                "is_paused": False,
            })

            # Refresh button (can be triggered automatically)
            refresh_queue_btn = gr.Button(
                "🔄 Refresh",
                size="sm",
                elem_id="refresh-queue-btn",
            )

    # ================== Helper Functions ==================

    def format_job_display(job_data: Dict[str, Any]) -> str:
        """Format a single job for display."""
        if not job_data:
            return ""

        job_type = job_data.get("job_type", "unknown")
        name = job_data.get("name", "Unnamed job")
        status = job_data.get("status", "unknown")
        job_id = job_data.get("id", "?")
        progress = job_data.get("progress", 0)
        progress_msg = job_data.get("progress_message", "")

        # Truncate name if too long
        if len(name) > 40:
            name = name[:37] + "..."

        display = f"[{job_id}] {name}"

        if status == "running":
            pct = int(progress * 100)
            display += f" ({pct}%)"
            if progress_msg:
                display += f" - {progress_msg}"

        return display

    def format_pending_jobs(pending_jobs: List[Dict[str, Any]]) -> str:
        """Format pending jobs list for display."""
        if not pending_jobs:
            return "No pending jobs"

        lines = []
        for i, job in enumerate(pending_jobs[:10]):  # Show max 10
            job_id = job.get("id", "?")
            name = job.get("name", "Unnamed")
            priority = job.get("priority", 2)

            # Truncate name
            if len(name) > 35:
                name = name[:32] + "..."

            priority_icon = {1: "🟢", 2: "🔵", 3: "🟡", 4: "🔴"}.get(priority, "⚪")
            lines.append(f"{i+1}. {priority_icon} [{job_id}] {name}")

        if len(pending_jobs) > 10:
            lines.append(f"... and {len(pending_jobs) - 10} more")

        return "\n".join(lines)

    def format_history(history: List[Dict[str, Any]]) -> str:
        """Format job history for display."""
        if not history:
            return "No completed jobs"

        lines = []
        for job in history[:20]:  # Show max 20
            job_id = job.get("id", "?")
            name = job.get("name", "Unnamed")
            status = job.get("status", "unknown")

            # Truncate name
            if len(name) > 30:
                name = name[:27] + "..."

            status_icon = {
                "completed": "✅",
                "failed": "❌",
                "cancelled": "⛔",
            }.get(status, "❓")

            lines.append(f"{status_icon} [{job_id}] {name}")

        return "\n".join(lines)

    def refresh_queue_display(state: Dict) -> Tuple[str, str, int, str, str, bool, bool, Dict]:
        """Refresh all queue displays."""
        try:
            status = get_queue_status_for_ui()

            # Format status summary
            pending_count = status.get("pending_count", 0)
            is_paused = status.get("is_paused", False)
            est_time = status.get("estimated_time_formatted", "")

            if is_paused:
                status_text = "**Queue Status**: ⏸️ Paused"
            elif status.get("active_job"):
                status_text = "**Queue Status**: 🔄 Processing"
            else:
                status_text = "**Queue Status**: Idle"

            if pending_count > 0:
                status_text += f"\n\n{pending_count} job(s) pending"
                if est_time:
                    status_text += f" (~{est_time})"

            # Format active job
            active_job = status.get("active_job")
            if active_job:
                active_display = format_job_display(active_job)
                active_progress = int(active_job.get("progress", 0) * 100)
                active_job_id = active_job.get("id")
                show_cancel_active = True
            else:
                active_display = "No active job"
                active_progress = 0
                active_job_id = None
                show_cancel_active = False

            # Format pending jobs
            pending_display = format_pending_jobs(status.get("pending_jobs", []))

            # Format history
            history_display_text = format_history(status.get("history", []))

            # Update state
            new_state = {
                "active_job_id": active_job_id,
                "pending_job_ids": [j.get("id") for j in status.get("pending_jobs", [])],
                "is_paused": is_paused,
            }

            return (
                status_text,
                active_display,
                active_progress,
                pending_display,
                history_display_text,
                not is_paused,  # pause_btn visible when not paused
                is_paused,      # resume_btn visible when paused
                new_state,
            )

        except Exception as e:
            print(f"Error refreshing queue display: {e}")
            return (
                "**Queue Status**: Error",
                f"Error: {str(e)}",
                0,
                "Error loading pending jobs",
                "Error loading history",
                True,
                False,
                state,
            )

    def pause_queue(state: Dict) -> Tuple[str, bool, bool, Dict]:
        """Pause the job queue."""
        try:
            manager.pause_queue()
            state["is_paused"] = True
            return "**Queue Status**: ⏸️ Paused", False, True, state
        except Exception as e:
            return f"Error: {e}", True, False, state

    def resume_queue(state: Dict) -> Tuple[str, bool, bool, Dict]:
        """Resume the job queue."""
        try:
            manager.resume_queue()
            state["is_paused"] = False
            return "**Queue Status**: 🔄 Processing", True, False, state
        except Exception as e:
            return f"Error: {e}", False, True, state

    def cancel_active_job(state: Dict) -> str:
        """Cancel the currently active job."""
        try:
            active_id = state.get("active_job_id")
            if active_id:
                manager.cancel_job(active_id)
                return "Cancellation requested..."
            return "No active job to cancel"
        except Exception as e:
            return f"Error: {e}"

    def cancel_selected_job(job_id: str) -> str:
        """Cancel a pending job by ID."""
        try:
            if job_id:
                success = manager.cancel_job(job_id.strip())
                if success:
                    return f"Job {job_id} cancelled"
                return f"Job {job_id} not found"
            return "No job ID provided"
        except Exception as e:
            return f"Error: {e}"

    def move_job_up(job_id: str) -> str:
        """Move a job up in the queue."""
        try:
            if job_id:
                success = manager.move_job_up(job_id.strip())
                if success:
                    return f"Job {job_id} moved up"
                return f"Cannot move job {job_id} up"
            return "No job ID provided"
        except Exception as e:
            return f"Error: {e}"

    def move_job_down(job_id: str) -> str:
        """Move a job down in the queue."""
        try:
            if job_id:
                success = manager.move_job_down(job_id.strip())
                if success:
                    return f"Job {job_id} moved down"
                return f"Cannot move job {job_id} down"
            return "No job ID provided"
        except Exception as e:
            return f"Error: {e}"

    def clear_pending_jobs() -> str:
        """Clear all pending jobs."""
        try:
            count = manager.clear_pending()
            return f"Cleared {count} pending job(s)"
        except Exception as e:
            return f"Error: {e}"

    def clear_job_history() -> str:
        """Clear job history."""
        try:
            count = manager.clear_history()
            return f"Cleared {count} history item(s)"
        except Exception as e:
            return f"Error: {e}"

    # ================== Wire Up Events ==================

    # Refresh button
    refresh_queue_btn.click(
        fn=refresh_queue_display,
        inputs=[queue_state],
        outputs=[
            queue_status_md,
            active_job_display,
            active_job_progress,
            pending_jobs_display,
            history_display,
            pause_btn,
            resume_btn,
            queue_state,
        ],
    )

    # Pause/Resume
    pause_btn.click(
        fn=pause_queue,
        inputs=[queue_state],
        outputs=[queue_status_md, pause_btn, resume_btn, queue_state],
    )

    resume_btn.click(
        fn=resume_queue,
        inputs=[queue_state],
        outputs=[queue_status_md, pause_btn, resume_btn, queue_state],
    )

    # Cancel active job
    cancel_active_btn.click(
        fn=cancel_active_job,
        inputs=[queue_state],
        outputs=[active_job_display],
    )

    # Clear pending
    clear_pending_btn.click(
        fn=clear_pending_jobs,
        outputs=[pending_jobs_display],
    )

    # Clear history
    clear_history_btn.click(
        fn=clear_job_history,
        outputs=[history_display],
    )

    # Job manipulation (hidden by default)
    move_up_btn.click(
        fn=move_job_up,
        inputs=[selected_job_id],
        outputs=[pending_jobs_display],
    )

    move_down_btn.click(
        fn=move_job_down,
        inputs=[selected_job_id],
        outputs=[pending_jobs_display],
    )

    cancel_job_btn.click(
        fn=cancel_selected_job,
        inputs=[selected_job_id],
        outputs=[pending_jobs_display],
    )

    return {
        "sidebar_column": sidebar_column,
        "queue_accordion": queue_accordion,
        "queue_status_md": queue_status_md,
        "active_job_display": active_job_display,
        "active_job_progress": active_job_progress,
        "pending_jobs_display": pending_jobs_display,
        "history_display": history_display,
        "queue_state": queue_state,
        "refresh_queue_btn": refresh_queue_btn,
        "pause_btn": pause_btn,
        "resume_btn": resume_btn,
        "cancel_active_btn": cancel_active_btn,
        "clear_pending_btn": clear_pending_btn,
        "clear_history_btn": clear_history_btn,
        "refresh_fn": refresh_queue_display,
    }


def get_queue_badge_count() -> str:
    """Get badge count for queue status (for display in header/toggle)."""
    try:
        manager = get_job_queue_manager()
        count = manager.get_pending_count()
        if manager.has_active_job():
            count += 1
        return str(count) if count > 0 else ""
    except Exception:
        return ""
