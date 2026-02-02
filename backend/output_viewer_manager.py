"""
MFLUX WebUI - Output Viewer Manager
Handles pagination, metadata extraction, caching, and search for the output viewer.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import hashlib
import threading

from backend.metadata_config_manager import MetadataConfigManager


@dataclass
class OutputItem:
    """Represents a single output image with its metadata."""
    filepath: str
    filename: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    file_size: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height
        }


class OutputViewerManager:
    """Manager for browsing and searching generated outputs."""

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp'}
    BATCH_SIZE = 20
    MEMORY_WINDOW_SIZE = 60
    THUMBNAIL_SIZE = (256, 256)

    def __init__(self, output_dir: str = "output"):
        # Make output_dir absolute relative to project root
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            # Get the project root (parent of backend directory)
            project_root = Path(__file__).parent.parent
            output_path = project_root / output_dir
        self.output_dir = output_path.resolve()
        self.metadata_manager = MetadataConfigManager()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._file_list_cache: Optional[List[Path]] = None
        self._file_list_cache_time: Optional[float] = None
        self._cache_lock = threading.Lock()
        self._thumbnail_dir = self.output_dir / ".thumbnails"

    def _ensure_thumbnail_dir(self):
        """Ensure thumbnail directory exists."""
        if not self._thumbnail_dir.exists():
            self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def _get_thumbnail_path(self, filepath: Path) -> Path:
        """Get the thumbnail path for an image."""
        # Use hash of filepath for thumbnail name to avoid conflicts
        file_hash = hashlib.md5(str(filepath).encode()).hexdigest()[:12]
        return self._thumbnail_dir / f"{file_hash}_thumb.jpg"

    def _generate_thumbnail(self, filepath: Path) -> Optional[str]:
        """Generate a thumbnail for an image if it doesn't exist."""
        self._ensure_thumbnail_dir()
        thumb_path = self._get_thumbnail_path(filepath)

        if thumb_path.exists():
            return str(thumb_path)

        try:
            with Image.open(filepath) as img:
                img.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                img.save(thumb_path, "JPEG", quality=85)
            return str(thumb_path)
        except Exception as e:
            print(f"Error generating thumbnail for {filepath}: {e}")
            return None

    def _get_sorted_files(self, force_refresh: bool = False) -> List[Path]:
        """Get list of output files sorted by modification time (newest first)."""
        with self._cache_lock:
            current_time = os.path.getmtime(self.output_dir) if self.output_dir.exists() else 0

            # Use cache if valid and not forcing refresh
            if (not force_refresh and
                self._file_list_cache is not None and
                self._file_list_cache_time == current_time):
                return self._file_list_cache

            if not self.output_dir.exists():
                self._file_list_cache = []
                self._file_list_cache_time = current_time
                return []

            files = []
            for file_path in self.output_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    files.append(file_path)

            # Sort by modification time, newest first
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            self._file_list_cache = files
            self._file_list_cache_time = current_time
            return files

    def get_output_count(self, search_query: str = "") -> int:
        """Get total count of outputs, optionally filtered by search."""
        files = self._get_sorted_files()

        if not search_query:
            return len(files)

        # Filter by search query (searches in prompt)
        count = 0
        search_lower = search_query.lower()
        for file_path in files:
            metadata = self.get_output_metadata(str(file_path))
            prompt = metadata.get("prompt", "")
            if search_lower in prompt.lower():
                count += 1
        return count

    def get_output_metadata(self, filepath: str) -> Dict[str, Any]:
        """Get metadata for a specific output file."""
        # Check cache first
        if filepath in self._metadata_cache:
            return self._metadata_cache[filepath]

        file_path = Path(filepath)
        metadata = {}

        try:
            # Extract metadata using the existing manager
            extracted = self.metadata_manager.extract_metadata_from_image(file_path)
            if extracted:
                # Map to standard fields
                metadata = {
                    "prompt": extracted.get("prompt", extracted.get("mflux_prompt", "")),
                    "model": extracted.get("model", extracted.get("mflux_model", "")),
                    "seed": extracted.get("seed", extracted.get("mflux_seed", "")),
                    "steps": extracted.get("steps", extracted.get("mflux_steps", "")),
                    "guidance": extracted.get("guidance", extracted.get("mflux_guidance", "")),
                    "width": extracted.get("width", extracted.get("mflux_width", "")),
                    "height": extracted.get("height", extracted.get("mflux_height", "")),
                    "lora_files": extracted.get("lora_files", extracted.get("mflux_lora_files", [])),
                    "lora_scales": extracted.get("lora_scales", extracted.get("mflux_lora_scales", [])),
                    "generation_time": extracted.get("generation_time", ""),
                    "generation_duration": extracted.get("generation_duration", ""),
                    "low_ram_mode": extracted.get("low_ram_mode", False),
                }

            # Get image dimensions if not in metadata
            if not metadata.get("width") or not metadata.get("height"):
                with Image.open(file_path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height

        except Exception as e:
            print(f"Error extracting metadata from {filepath}: {e}")

        # Cache the metadata
        self._metadata_cache[filepath] = metadata
        return metadata

    def get_outputs_paginated(
        self,
        offset: int = 0,
        limit: int = None,
        search_query: str = ""
    ) -> Tuple[List[OutputItem], int]:
        """
        Get a paginated list of outputs.

        Args:
            offset: Starting index
            limit: Number of items to return (defaults to BATCH_SIZE)
            search_query: Optional search string to filter by prompt

        Returns:
            Tuple of (list of OutputItems, total count matching query)
        """
        if limit is None:
            limit = self.BATCH_SIZE

        files = self._get_sorted_files()

        # Filter by search query if provided
        if search_query:
            search_lower = search_query.lower()
            filtered_files = []
            for file_path in files:
                metadata = self.get_output_metadata(str(file_path))
                prompt = metadata.get("prompt", "")
                if search_lower in prompt.lower():
                    filtered_files.append(file_path)
            files = filtered_files

        total_count = len(files)

        # Apply pagination
        paginated_files = files[offset:offset + limit]

        items = []
        for file_path in paginated_files:
            metadata = self.get_output_metadata(str(file_path))
            stat = file_path.stat()

            item = OutputItem(
                filepath=str(file_path),
                filename=file_path.name,
                metadata=metadata,
                created_at=datetime.fromtimestamp(stat.st_mtime),
                file_size=stat.st_size,
                width=metadata.get("width", 0),
                height=metadata.get("height", 0)
            )
            items.append(item)

        return items, total_count

    def get_outputs_for_gallery(
        self,
        offset: int = 0,
        limit: int = None,
        search_query: str = "",
        use_thumbnails: bool = True
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], int]:
        """
        Get outputs formatted for Gradio Gallery.

        Returns:
            Tuple of (
                list of (image_path, caption) tuples for gallery,
                list of metadata dicts,
                total count
            )
        """
        try:
            items, total_count = self.get_outputs_paginated(offset, limit, search_query)
        except Exception as e:
            print(f"Error getting paginated outputs: {e}")
            return [], [], 0

        gallery_items = []
        metadata_list = []

        for item in items:
            try:
                # Use thumbnail if available and requested
                if use_thumbnails:
                    thumb_path = self._generate_thumbnail(Path(item.filepath))
                    image_path = thumb_path if thumb_path else item.filepath
                else:
                    image_path = item.filepath

                # Create caption from prompt (truncated)
                prompt = item.metadata.get("prompt", "No prompt")
                caption = prompt[:50] + "..." if len(prompt) > 50 else prompt

                gallery_items.append((image_path, caption))
                metadata_list.append({
                    "filepath": item.filepath,
                    "filename": item.filename,
                    **item.metadata,
                    "created_at": item.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": item.file_size
                })
            except Exception as e:
                print(f"Error processing output item {item.filepath}: {e}")
                continue

        return gallery_items, metadata_list, total_count

    def search_outputs(self, query: str) -> List[OutputItem]:
        """Search outputs by prompt text."""
        items, _ = self.get_outputs_paginated(search_query=query, limit=1000)
        return items

    def delete_output(self, filepath: str) -> bool:
        """Delete an output file and its thumbnail."""
        try:
            file_path = Path(filepath)

            # Delete thumbnail if exists
            thumb_path = self._get_thumbnail_path(file_path)
            if thumb_path.exists():
                thumb_path.unlink()

            # Delete the file
            if file_path.exists():
                file_path.unlink()

            # Remove from caches
            if filepath in self._metadata_cache:
                del self._metadata_cache[filepath]

            # Invalidate file list cache
            self._file_list_cache = None

            return True
        except Exception as e:
            print(f"Error deleting {filepath}: {e}")
            return False

    def refresh_cache(self):
        """Force refresh of all caches."""
        with self._cache_lock:
            self._file_list_cache = None
            self._file_list_cache_time = None
        self._metadata_cache.clear()

    def get_settings_json(self, filepath: str) -> str:
        """Get generation settings as JSON string for copying."""
        metadata = self.get_output_metadata(filepath)
        settings = {
            "prompt": metadata.get("prompt", ""),
            "model": metadata.get("model", ""),
            "seed": metadata.get("seed", ""),
            "steps": metadata.get("steps", ""),
            "guidance": metadata.get("guidance", ""),
            "width": metadata.get("width", ""),
            "height": metadata.get("height", ""),
            "lora_files": metadata.get("lora_files", []),
            "lora_scales": metadata.get("lora_scales", []),
        }
        return json.dumps(settings, indent=2)


# Global instance
_output_viewer_manager: Optional[OutputViewerManager] = None


def get_output_viewer_manager() -> OutputViewerManager:
    """Get the global OutputViewerManager instance."""
    global _output_viewer_manager
    if _output_viewer_manager is None:
        _output_viewer_manager = OutputViewerManager()
    return _output_viewer_manager
