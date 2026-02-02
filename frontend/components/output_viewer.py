"""
MFLUX WebUI - Output Viewer Component
Displays all generated outputs with lazy loading, search, and metadata viewing.
"""

import gradio as gr
import json
from typing import List, Tuple, Dict, Any, Optional

from backend.output_viewer_manager import get_output_viewer_manager, OutputViewerManager


# Constants
BATCH_SIZE = 20
MEMORY_WINDOW = 60  # Keep 60 images max in gallery
PREFETCH_THRESHOLD = 0.7  # Load more at 70% scroll


def create_output_viewer_tab():
    """Create the Output Viewer tab interface."""

    manager = get_output_viewer_manager()

    # JavaScript for scroll detection and lazy loading
    scroll_detection_js = """
    () => {
        // Wait for gallery to be ready
        setTimeout(() => {
            const gallery = document.querySelector('#output-viewer-gallery .grid-wrap');
            if (!gallery) return;

            let isLoading = false;
            let lastScrollTop = 0;

            gallery.addEventListener('scroll', () => {
                if (isLoading) return;

                const scrollTop = gallery.scrollTop;
                const scrollHeight = gallery.scrollHeight;
                const clientHeight = gallery.clientHeight;

                if (scrollHeight <= clientHeight) return;

                const scrollPercent = scrollTop / (scrollHeight - clientHeight);

                // Scrolling down and past threshold - load more
                if (scrollTop > lastScrollTop && scrollPercent > 0.7) {
                    isLoading = true;
                    const loadBtn = document.querySelector('#load-more-btn');
                    if (loadBtn) {
                        loadBtn.click();
                        setTimeout(() => { isLoading = false; }, 500);
                    }
                }

                lastScrollTop = scrollTop;
            });
        }, 1000);
    }
    """

    with gr.TabItem("Output Viewer"):
        # State for tracking loaded data
        viewer_state = gr.State({
            "offset": 0,
            "total_count": 0,
            "search_query": "",
            "all_metadata": [],
            "selected_index": -1,
            "gallery_items": [],
        })

        with gr.Row():
            with gr.Column(scale=3):
                # Search and controls
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search prompts...",
                        placeholder="Type to search by prompt text",
                        scale=8
                    )
                    refresh_btn = gr.Button(
                        "Refresh",
                        variant="secondary",
                        scale=1
                    )

                # Output count display
                output_count_display = gr.Markdown("Click **Refresh** to load outputs")

                # Gallery for displaying outputs
                output_gallery = gr.Gallery(
                    label="Generated Outputs",
                    show_label=False,
                    elem_id="output-viewer-gallery",
                    columns=4,
                    rows=3,
                    object_fit="cover",
                    height=500,
                    allow_preview=True,
                    preview=False  # Disabled preview mode to fix loading issues
                )

                # Hidden button for triggering load more
                load_more_btn = gr.Button(
                    "Load More",
                    visible=True,
                    elem_id="load-more-btn",
                    size="sm"
                )

            with gr.Column(scale=2):
                # Selected image details panel
                gr.Markdown("### Selected Image Details")

                selected_image = gr.Image(
                    label="Selected Image",
                    show_label=False,
                    height=300,
                    interactive=False
                )

                # Metadata display
                with gr.Group():
                    prompt_display = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        interactive=False
                    )

                    with gr.Row():
                        model_display = gr.Textbox(
                            label="Model",
                            interactive=False,
                            scale=2
                        )
                        seed_display = gr.Textbox(
                            label="Seed",
                            interactive=False,
                            scale=1
                        )

                    with gr.Row():
                        steps_display = gr.Textbox(
                            label="Steps",
                            interactive=False
                        )
                        guidance_display = gr.Textbox(
                            label="Guidance",
                            interactive=False
                        )
                        dimensions_display = gr.Textbox(
                            label="Dimensions",
                            interactive=False
                        )

                    lora_display = gr.Textbox(
                        label="LoRA Files",
                        interactive=False,
                        visible=True
                    )

                    generated_at_display = gr.Textbox(
                        label="Generated At",
                        interactive=False
                    )

                # Action buttons
                with gr.Row():
                    copy_settings_btn = gr.Button(
                        "Copy Settings",
                        variant="secondary",
                        size="sm"
                    )
                    delete_btn = gr.Button(
                        "Delete",
                        variant="stop",
                        size="sm"
                    )

                # Hidden field to store selected filepath
                selected_filepath = gr.Textbox(visible=False)

                # Copy result display
                copy_result = gr.Textbox(
                    label="Settings JSON (select all and copy)",
                    lines=6,
                    visible=False
                )

                # Delete confirmation
                with gr.Row(visible=False) as delete_confirm_row:
                    gr.Markdown("Are you sure you want to delete this image?")
                    confirm_delete_btn = gr.Button("Yes, Delete", variant="stop", size="sm")
                    cancel_delete_btn = gr.Button("Cancel", size="sm")

    # Helper functions
    def load_initial_outputs(state):
        """Load initial batch of outputs."""
        try:
            gallery_items, metadata_list, total = manager.get_outputs_for_gallery(
                offset=0,
                limit=BATCH_SIZE,
                search_query=""
            )

            new_state = {
                "offset": len(gallery_items),
                "total_count": total,
                "search_query": "",
                "all_metadata": metadata_list,
                "selected_index": -1,
                "gallery_items": gallery_items,
            }

            count_text = f"Showing {len(gallery_items)} of {total} outputs"

            return gallery_items, new_state, count_text
        except Exception as e:
            print(f"Error loading initial outputs: {e}")
            return [], {"offset": 0, "total_count": 0, "search_query": "", "all_metadata": [], "selected_index": -1, "gallery_items": []}, "Error loading outputs"

    def load_more_outputs(state):
        """Load next batch of outputs (for infinite scroll)."""
        current_offset = state.get("offset", 0)
        search_query = state.get("search_query", "")
        current_metadata = state.get("all_metadata", [])
        current_gallery = state.get("gallery_items", [])

        # Check if there's more to load
        total = manager.get_output_count(search_query)
        if current_offset >= total:
            count_text = f"Showing {len(current_metadata)} of {total} outputs"
            return gr.update(), state, count_text

        # Load next batch (uses thumbnails via get_outputs_for_gallery)
        new_gallery_items, metadata_list, _ = manager.get_outputs_for_gallery(
            offset=current_offset,
            limit=BATCH_SIZE,
            search_query=search_query
        )

        if not new_gallery_items:
            count_text = f"Showing {len(current_metadata)} of {total} outputs"
            return gr.update(), state, count_text

        # Combine with existing
        all_metadata = current_metadata + metadata_list
        combined_gallery = current_gallery + new_gallery_items

        # Memory management: keep only last MEMORY_WINDOW items if we exceed
        if len(all_metadata) > MEMORY_WINDOW:
            trim_count = len(all_metadata) - MEMORY_WINDOW
            all_metadata = all_metadata[trim_count:]
            combined_gallery = combined_gallery[trim_count:]

        new_state = {
            **state,
            "offset": current_offset + len(new_gallery_items),
            "all_metadata": all_metadata,
            "gallery_items": combined_gallery,
        }

        count_text = f"Showing {len(all_metadata)} of {total} outputs"

        return combined_gallery, new_state, count_text

    def search_outputs(query, state):
        """Search outputs by prompt text."""
        manager.refresh_cache()  # Refresh to get latest

        gallery_items, metadata_list, total = manager.get_outputs_for_gallery(
            offset=0,
            limit=BATCH_SIZE,
            search_query=query
        )

        new_state = {
            "offset": len(gallery_items),
            "total_count": total,
            "search_query": query,
            "all_metadata": metadata_list,
            "selected_index": -1,
            "gallery_items": gallery_items,
        }

        if query:
            count_text = f"Found {total} outputs matching '{query}'"
        else:
            count_text = f"Showing {len(gallery_items)} of {total} outputs"

        return gallery_items, new_state, count_text

    def refresh_outputs(state):
        """Refresh the output list."""
        from pathlib import Path

        search_query = state.get("search_query", "")

        try:
            # Direct file listing for reliability
            output_dir = Path(__file__).parent.parent.parent / "output"

            if not output_dir.exists():
                return [], state, "No output directory found"

            # Get image files
            supported = {'.png', '.jpg', '.jpeg', '.webp'}
            files = [f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() in supported]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            total = len(files)

            # Return file paths directly (simplest format)
            gallery_items = [str(f) for f in files[:BATCH_SIZE]]

            new_state = {
                "offset": len(gallery_items),
                "total_count": total,
                "search_query": search_query,
                "all_metadata": [],
                "selected_index": -1,
                "gallery_items": gallery_items,
            }

            count_text = f"Showing {len(gallery_items)} of {total} outputs"

            return gallery_items, new_state, count_text

        except Exception as e:
            print(f"Error refreshing outputs: {e}")
            return [], state, f"Error: {e}"

    def on_gallery_select(evt: gr.SelectData, state):
        """Handle gallery image selection."""
        if evt.index is None:
            return (
                None, "", "", "", "", "", "", "", "",
                gr.update(visible=False), state
            )

        all_metadata = state.get("all_metadata", [])

        if evt.index >= len(all_metadata):
            return (
                None, "", "", "", "", "", "", "", "",
                gr.update(visible=False), state
            )

        meta = all_metadata[evt.index]
        filepath = meta.get("filepath", "")

        # Format LoRA display
        lora_files = meta.get("lora_files", [])
        lora_scales = meta.get("lora_scales", [])
        if lora_files and isinstance(lora_files, list):
            if isinstance(lora_files, str):
                try:
                    lora_files = json.loads(lora_files)
                except:
                    lora_files = [lora_files]
            if isinstance(lora_scales, str):
                try:
                    lora_scales = json.loads(lora_scales)
                except:
                    lora_scales = []

            lora_text = []
            for i, lora in enumerate(lora_files):
                scale = lora_scales[i] if i < len(lora_scales) else 1.0
                lora_name = lora.split("/")[-1] if "/" in str(lora) else str(lora)
                lora_text.append(f"{lora_name} (scale: {scale})")
            lora_display_text = "\n".join(lora_text) if lora_text else "None"
        else:
            lora_display_text = "None"

        # Format dimensions
        width = meta.get("width", "?")
        height = meta.get("height", "?")
        dimensions = f"{width} x {height}"

        new_state = {**state, "selected_index": evt.index}

        return (
            filepath,  # selected_image
            meta.get("prompt", ""),  # prompt_display
            meta.get("model", ""),  # model_display
            str(meta.get("seed", "")),  # seed_display
            str(meta.get("steps", "")),  # steps_display
            str(meta.get("guidance", "")),  # guidance_display
            dimensions,  # dimensions_display
            lora_display_text,  # lora_display
            meta.get("created_at", ""),  # generated_at_display
            filepath,  # selected_filepath
            new_state
        )

    def copy_settings(filepath):
        """Copy generation settings to clipboard."""
        if not filepath:
            return gr.update(visible=False), ""

        settings_json = manager.get_settings_json(filepath)
        return gr.update(visible=True), settings_json

    def show_delete_confirm():
        """Show delete confirmation."""
        return gr.update(visible=True)

    def hide_delete_confirm():
        """Hide delete confirmation."""
        return gr.update(visible=False)

    def delete_output(filepath, state):
        """Delete the selected output."""
        if not filepath:
            return gr.update(visible=False), gr.update(), state, ""

        success = manager.delete_output(filepath)

        if success:
            # Refresh the gallery
            search_query = state.get("search_query", "")
            gallery_items, metadata_list, total = manager.get_outputs_for_gallery(
                offset=0,
                limit=BATCH_SIZE,
                search_query=search_query
            )

            new_state = {
                "offset": len(gallery_items),
                "total_count": total,
                "search_query": search_query,
                "all_metadata": metadata_list,
                "selected_index": -1,
                "gallery_items": gallery_items,
            }

            count_text = f"Showing {len(gallery_items)} of {total} outputs"

            return (
                gr.update(visible=False),  # hide confirm
                gallery_items,  # update gallery
                new_state,
                count_text
            )
        else:
            return gr.update(visible=False), gr.update(), state, gr.update()

    # Wire up events
    # Note: Initial load is handled via demo.load() in gradioui.py

    # Search functionality
    search_box.change(
        fn=search_outputs,
        inputs=[search_box, viewer_state],
        outputs=[output_gallery, viewer_state, output_count_display]
    )

    # Refresh button
    refresh_btn.click(
        fn=refresh_outputs,
        inputs=[viewer_state],
        outputs=[output_gallery, viewer_state, output_count_display]
    )

    # Load more button (triggered by scroll or manually)
    load_more_btn.click(
        fn=load_more_outputs,
        inputs=[viewer_state],
        outputs=[output_gallery, viewer_state, output_count_display]
    )

    # Gallery selection
    output_gallery.select(
        fn=on_gallery_select,
        inputs=[viewer_state],
        outputs=[
            selected_image,
            prompt_display,
            model_display,
            seed_display,
            steps_display,
            guidance_display,
            dimensions_display,
            lora_display,
            generated_at_display,
            selected_filepath,
            viewer_state
        ]
    )

    # Copy settings
    copy_settings_btn.click(
        fn=copy_settings,
        inputs=[selected_filepath],
        outputs=[copy_result, copy_result]
    )

    # Delete flow
    delete_btn.click(
        fn=show_delete_confirm,
        outputs=[delete_confirm_row]
    )

    cancel_delete_btn.click(
        fn=hide_delete_confirm,
        outputs=[delete_confirm_row]
    )

    confirm_delete_btn.click(
        fn=delete_output,
        inputs=[selected_filepath, viewer_state],
        outputs=[delete_confirm_row, output_gallery, viewer_state, output_count_display]
    )

    return {
        "gallery": output_gallery,
        "viewer_state": viewer_state,
        "output_count_display": output_count_display,
        "search_box": search_box,
        "refresh_btn": refresh_btn,
        "load_more_btn": load_more_btn,
        "load_initial": load_initial_outputs,
        "scroll_js": scroll_detection_js,
    }
