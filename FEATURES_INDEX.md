# MFLUX WebUI - Features Index & AI Agent Guide

> **Purpose**: This document provides a comprehensive index of all features in MFLUX WebUI v0.15.4, mapping them to their implementation files. Use this as a quick reference guide for AI agents to locate and work with specific features.

---

## Table of Contents
1. [Core Generation Features](#core-generation-features)
2. [Advanced Generation Features](#advanced-generation-features)
3. [Configuration & Management](#configuration--management)
4. [Integration Features](#integration-features)
5. [Utility & Support Features](#utility--support-features)
6. [Frontend Components](#frontend-components)
7. [Prompt Management](#prompt-management)

---

## Core Generation Features

### Text-to-Image Generation (Basic)
**Description**: Simple text-to-image generation with MFLUX models  
**Backend Files**:
- `backend/flux_manager.py` - Main Flux generation logic (`generate_image_gradio`, `generate_image_batch`)
- `backend/image_generation.py` - Image generation utilities

**Frontend Files**:
- `frontend/components/easy_mflux.py` - Simple UI tab (`create_easy_mflux_tab`)

**Key Functions**:
- `simple_generate_image()` - Quick generation wrapper
- `generate_image_gradio()` - Full generation with all parameters

---

### Text-to-Image Generation (Advanced)
**Description**: Advanced text-to-image with full control over parameters  
**Backend Files**:
- `backend/flux_manager.py` - Advanced generation logic
- `backend/generation_workflow.py` - Workflow management with dynamic prompts and auto-seeds

**Frontend Files**:
- `frontend/components/advanced_generate.py` - Advanced UI tab (`create_advanced_generate_tab`)

**Key Functions**:
- `generate_image_gradio()` - Full parameter control
- `get_generation_workflow()` - Workflow integration

---

### Image-to-Image (img2img)
**Description**: Transform existing images using prompts and MFLUX models  
**Backend Files**:
- `backend/flux_manager.py` - Image-to-image generation (`generate_image_i2i_gradio`)

**Frontend Files**:
- `frontend/components/image_to_image.py` - img2img UI tab (`create_image_to_image_tab`)

**Key Functions**:
- `generate_image_i2i_gradio()` - Image transformation with init_image support

---

### Fill Tool (Inpaint/Outpaint)
**Description**: Inpainting and outpainting using FLUX.1-Fill-dev  
**Backend Files**:
- `backend/fill_manager.py` - Fill/inpaint/outpaint logic (`generate_fill_gradio`)

**Frontend Files**:
- `frontend/components/fill.py` - Fill UI tab (`create_fill_tab`)

**Key Functions**:
- `get_or_create_flux_fill()` - Flux Fill model initialization
- `generate_fill_gradio()` - Fill generation with mask support

---

### Depth-Guided Generation
**Description**: Generate images guided by depth maps  
**Backend Files**:
- `backend/depth_manager.py` - Depth map generation and depth-guided generation
- `backend/depth_export_manager.py` - Depth map export utilities

**Frontend Files**:
- `frontend/components/depth.py` - Depth UI tab (`create_depth_tab`)

**Key Functions**:
- `generate_depth_map()` - Create depth maps from images
- `get_or_create_flux_depth()` - Depth-aware Flux model

---

### ControlNet
**Description**: ControlNet-based generation with various control types  
**Backend Files**:
- `backend/flux_manager.py` - ControlNet generation (`generate_image_controlnet_gradio`)

**Frontend Files**:
- `frontend/components/controlnet.py` - ControlNet UI tab (`create_controlnet_tab`)

**Key Functions**:
- `generate_image_controlnet_gradio()` - ControlNet generation

---

### Flux2 Klein (4B/9B)
**Description**: Next-generation Flux2 Klein models for text-to-image and editing  
**Backend Files**:
- `backend/flux2_manager.py` - Flux2 Klein generation (`generate_flux2_image_gradio`, `generate_flux2_edit_gradio`)

**Frontend Files**:
- `frontend/components/flux2_generate.py` - Flux2 generation tab (`create_flux2_generate_tab`)
- `frontend/components/flux2_edit.py` - Flux2 edit tab (`create_flux2_edit_tab`)

**Key Functions**:
- `generate_flux2_image_gradio()` - Flux2 text-to-image
- `generate_flux2_edit_gradio()` - Flux2 multi-image editing

---

### Qwen Image & Qwen Edit
**Description**: Multilingual text-to-image generation and image editing with Qwen models  
**Backend Files**:
- `backend/qwen_manager.py` - Qwen generation (`generate_qwen_image_gradio`, `generate_qwen_edit_gradio`)

**Frontend Files**:
- `frontend/components/qwen_image.py` - Qwen image generation tab (`create_qwen_image_tab`)
- `frontend/components/qwen_edit.py` - Qwen edit tab (`create_qwen_edit_tab`)

**Key Functions**:
- `generate_qwen_image_gradio()` - Qwen text-to-image with multilingual support
- `generate_qwen_edit_gradio()` - Qwen-based image editing

---

### FIBO (Structured Prompts)
**Description**: Structured prompt generation with JSON input and optional VLM expansion  
**Backend Files**:
- `backend/fibo_manager.py` - FIBO generation (`generate_fibo_gradio`)

**Frontend Files**:
- `frontend/components/fibo.py` - FIBO UI tab (`create_fibo_tab`)

**Key Functions**:
- `generate_fibo_gradio()` - FIBO structured prompt generation

---

### Z-Image Turbo
**Description**: Fast text-to-image generation with Z-Image Turbo model  
**Backend Files**:
- `backend/z_image_manager.py` - Z-Image generation (`generate_z_image_gradio`)

**Frontend Files**:
- `frontend/components/z_image_turbo.py` - Z-Image UI tab (`create_z_image_turbo_tab`)

**Key Functions**:
- `generate_z_image_gradio()` - Fast turbo generation with LoRA and img2img support

---

### Redux (Image Variations)
**Description**: Generate variations of existing images  
**Backend Files**:
- `backend/redux_manager.py` - Redux generation (`generate_redux_gradio`)

**Frontend Files**:
- `frontend/components/redux.py` - Redux UI tab (`create_redux_tab`)

**Key Functions**:
- `get_or_create_flux_redux()` - Redux model initialization
- `generate_redux_gradio()` - Image variation generation

---

### Upscaling
**Description**: High-resolution upscaling with ControlNet-aware upscaler  
**Backend Files**:
- `backend/upscale_manager.py` - Upscaling logic (`upscale_image_gradio`)
- `backend/seedvr2_manager.py` - SeedVR2 1-step upscaling (`generate_seedvr2_upscale`)

**Frontend Files**:
- `frontend/components/upscale.py` - Upscale UI tab (`create_upscale_tab`)

**Key Functions**:
- `upscale_image_gradio()` - ControlNet upscaler
- `generate_seedvr2_upscale()` - SeedVR2 faithful upscaling with softness control

---

### CatVTON (Virtual Try-On)
**Description**: Virtual clothing try-on using CatVTON  
**Backend Files**:
- `backend/catvton_manager.py` - CatVTON generation (`generate_catvton_gradio`)

**Frontend Files**:
- `frontend/components/catvton.py` - CatVTON UI tab (`create_catvton_tab`)

**Key Functions**:
- `get_or_create_flux_catvton()` - CatVTON model initialization
- `generate_catvton_gradio()` - Virtual try-on generation

---

### IC-Edit (In-Context Editing)
**Description**: In-context editing with example-based transformations  
**Backend Files**:
- `backend/ic_edit_manager.py` - IC-Edit generation (`generate_ic_edit_gradio`)

**Frontend Files**:
- `frontend/components/ic_edit.py` - IC-Edit UI tab (`create_ic_edit_tab`)

**Key Functions**:
- `get_or_create_flux_ic_edit()` - IC-Edit model initialization
- `generate_ic_edit_gradio()` - In-context editing

---

### Concept Attention
**Description**: Fine-grained prompt control with attention heatmaps  
**Backend Files**:
- `backend/concept_attention_manager.py` - Concept attention (`generate_text_concept_heatmap`, `generate_image_concept_heatmap`)

**Frontend Files**:
- `frontend/components/concept_attention.py` - Concept attention UI tab (`create_concept_attention_tab`)

**Key Functions**:
- `generate_text_concept_heatmap()` - Text-based concept focusing
- `generate_image_concept_heatmap()` - Image-based concept focusing

---

### Kontext
**Description**: Context-aware image editing  
**Backend Files**:
- `backend/kontext_manager.py` - Kontext generation (`generate_image_kontext_gradio`)
- `backend/flux_manager.py` - Legacy kontext support

**Frontend Files**:
- `frontend/components/kontext.py` - Kontext UI tab (`create_kontext_tab`)

**Key Functions**:
- `generate_image_kontext_gradio()` - Kontext editing

---

### Canvas (Prompt Enhancement)
**Description**: Visual canvas for prompt construction and enhancement  
**Backend Files**:
- Uses standard generation with enhanced prompts

**Frontend Files**:
- `frontend/components/canvas.py` - Canvas UI tab (`create_canvas_tab`)

**Prompts**:
- `prompts/canvas/canvas_generation_prompt.md`

---

### In-Context LoRA
**Description**: Dynamic LoRA application during generation  
**Backend Files**:
- `backend/flux_manager.py` - In-context LoRA generation (`generate_image_in_context_lora_gradio`)

**Frontend Files**:
- `frontend/components/in_context_lora.py` - In-context LoRA UI tab (`create_in_context_lora_tab`)

**Key Functions**:
- `generate_image_in_context_lora_gradio()` - Dynamic LoRA generation

---

## Advanced Generation Features

### Dynamic Prompts
**Description**: Wildcard support and prompt variations using [option1|option2] syntax  
**Backend Files**:
- `backend/dynamic_prompts_manager.py` - Dynamic prompt processing (`DynamicPromptsManager`)
- `backend/generation_workflow.py` - Workflow integration

**Frontend Files**:
- `frontend/components/dynamic_prompts.py` - Dynamic prompts UI tab (`create_dynamic_prompts_tab`)

**Configuration**:
- `prompts/dynamic_prompts_config.json` - Dynamic prompts configuration

**Key Classes**:
- `DynamicPromptsManager` - Manages wildcards and variations

**Key Functions**:
- `get_random_prompt_variation()` - Process dynamic prompts
- `process_prompt()` - Workflow integration

---

### Auto Seeds
**Description**: Intelligent seed management and selection for reproducible generation  
**Backend Files**:
- `backend/auto_seeds_manager.py` - Auto seeds logic (`AutoSeedsManager`)
- `backend/generation_workflow.py` - Workflow integration

**Frontend Files**:
- `frontend/components/auto_seeds.py` - Auto seeds UI tab (`create_auto_seeds_tab`)

**Configuration**:
- `configs/auto_seeds.json` - Seed pool configuration

**Key Classes**:
- `AutoSeedsManager` - Manages seed pools

**Key Functions**:
- `get_next_seed()` - Get next seed from pool
- `generate_seed_pool()` - Create seed collections

---

### Generation Workflow
**Description**: Comprehensive workflow management with progress tracking and statistics  
**Backend Files**:
- `backend/generation_workflow.py` - Main workflow manager (`GenerationWorkflow`)

**Key Classes**:
- `GenerationWorkflow` - Orchestrates generation features

**Key Functions**:
- `pre_generation_checks()` - Validation before generation
- `process_prompt()` - Dynamic prompt processing
- `get_seed_for_generation()` - Auto-seeds integration
- `monitor_generation_progress()` - Progress tracking
- `save_enhanced_metadata()` - Metadata management
- `update_generation_stats()` - Statistics tracking

---

### Stepwise Output
**Description**: Progressive image output during generation for visualization  
**Backend Files**:
- `backend/stepwise_output_manager.py` - Stepwise output manager (`StepwiseOutputManager`)

**Key Classes**:
- `StepwiseOutputManager` - Manages step-by-step output

**Key Functions**:
- `setup_stepwise_output()` - Configure stepwise output
- `save_step_image()` - Save intermediate steps

---

## Configuration & Management

### Configuration Manager
**Description**: Advanced configuration handling with presets and validation  
**Backend Files**:
- `backend/config_manager.py` - Configuration management (`ConfigManager`)

**Frontend Files**:
- `frontend/components/config_manager.py` - Config UI tab (`create_config_tab`)

**Configuration**:
- `configs/default_config.json` - Default configuration

**Key Classes**:
- `ConfigManager` - Manages app configurations

**Key Functions**:
- `load_config()` - Load configuration from file
- `save_config()` - Save configuration
- `validate_config()` - Configuration validation
- `get_preset()` - Load configuration preset

---

### Model Manager
**Description**: Model downloading, management, and quantization  
**Backend Files**:
- `backend/model_manager.py` - Model management (`CustomModelConfig`)

**Frontend Files**:
- `frontend/components/model_lora_management.py` - Model/LoRA management UI (`create_model_lora_management_tab`)

**Key Classes**:
- `CustomModelConfig` - Custom model configuration

**Key Functions**:
- `get_updated_models()` - List available models
- `get_custom_model_config()` - Get model configuration
- `download_and_save_model()` - Download models from HuggingFace
- `resolve_local_path()` - Resolve model paths
- `normalize_base_model_choice()` - Normalize model names
- `strip_quant_suffix()` - Remove quantization suffix

---

### LoRA Manager
**Description**: LoRA file management, downloading, and conversion  
**Backend Files**:
- `backend/lora_manager.py` - LoRA management functions
- `backend/lora_converter.py` - LoRA format conversion

**Frontend Files**:
- `frontend/components/model_lora_management.py` - LoRA UI

**Key Functions**:
- `get_available_lora_files()` - List available LoRA files
- `get_lora_choices()` - Get LoRA dropdown choices
- `process_lora_files()` - Process LoRA file paths
- `download_lora()` - Download LoRA from URL
- `refresh_lora_choices()` - Refresh LoRA list

**Environment Variables**:
- `LORA_LIBRARY_PATH` - External LoRA library paths

---

### Training Manager (Dreambooth)
**Description**: Dreambooth fine-tuning for custom models  
**Backend Files**:
- `backend/training_manager.py` - Training orchestration
- `backend/custom_train.py` - Custom training logic
- `backend/custom_trainer.py` - Trainer implementation

**Frontend Files**:
- `frontend/components/dreambooth_fine_tuning.py` - Training UI tab (`create_dreambooth_fine_tuning_tab`)

**Configuration**:
- `configs/` - Training configuration files

**Key Functions**:
- `prepare_training_config()` - Setup training parameters
- `run_training()` - Execute training process
- `run_dreambooth_from_ui_no_explicit_quantize()` - UI-triggered training

---

### Metadata Manager
**Description**: Enhanced metadata export and configuration from metadata  
**Backend Files**:
- `backend/metadata_config_manager.py` - Metadata management (`MetadataConfigManager`)

**Key Classes**:
- `MetadataConfigManager` - Load config from image metadata

**Key Functions**:
- `extract_metadata_from_image()` - Read metadata from images
- `config_from_metadata()` - Generate config from metadata
- `save_config_to_metadata()` - Embed config in images

---

### Export Manager
**Description**: Model export and quantization utilities  
**Backend Files**:
- `backend/export_manager.py` - Model export functions

**Key Functions**:
- `quantize_model()` - Quantize models (3,4,6,8-bit)
- `export_model()` - Export models with quantization

---

### VAE Tiling Manager
**Description**: Handle large images with VAE tiling  
**Backend Files**:
- `backend/vae_tiling_manager.py` - VAE tiling logic (`VAETilingManager`)

**Key Classes**:
- `VAETilingManager` - Manages VAE tiling for large images

**Key Functions**:
- `setup_vae_tiling()` - Configure tiling parameters
- `calculate_tiles()` - Compute tile positions
- `process_tiled()` - Process image with tiling

---

### Settings Manager
**Description**: LLM settings persistence across tabs  
**Backend Files**:
- `backend/settings_manager.py` - Settings storage

**Configuration**:
- `llm_settings.json` - LLM settings file

**Key Functions**:
- `load_llm_settings()` - Load settings for tab
- `save_llm_settings()` - Save settings for tab

---

## Integration Features

### Ollama Integration
**Description**: Prompt enhancement using local Ollama models  
**Backend Files**:
- `backend/ollama_manager.py` - Ollama integration

**Frontend Files**:
- `frontend/components/llmsettings.py` - LLM settings UI (`create_llm_settings`)

**Key Functions**:
- `get_available_ollama_models()` - List Ollama models
- `enhance_prompt()` - Enhance prompts with Ollama
- `ensure_llama_model()` - Download Ollama model

---

### MLX-VLM Integration
**Description**: Vision-language models for image captioning and analysis  
**Backend Files**:
- `backend/mlx_vlm_manager.py` - MLX-VLM integration
- `backend/captions.py` - Image captioning

**Key Classes**:
- `VisionModelWrapper` - MLX-VLM model wrapper
- `SimpleKVCache` - KV cache for VLM

**Key Functions**:
- `get_available_mlx_vlm_models()` - List MLX-VLM models
- `generate_caption_with_mlx_vlm()` - Generate image captions
- `load_mlx_model()` - Load MLX-VLM model
- `enhance_prompt_with_mlx()` - Enhance prompts with VLM

---

### HuggingFace Integration
**Description**: Download models and LoRAs from HuggingFace  
**Backend Files**:
- `backend/huggingface_manager.py` - HuggingFace integration

**Frontend Files**:
- `frontend/components/model_lora_management.py` - HF download UI

**Key Functions**:
- `login_huggingface()` - Authenticate with HF
- `download_and_save_model()` - Download HF models
- `download_lora_model_huggingface()` - Download HF LoRAs
- `get_available_models()` - List HF models
- `load_api_key()` / `save_api_key()` - API key management

---

### CivitAI Integration
**Description**: Download LoRA models from CivitAI marketplace  
**Backend Files**:
- `backend/civitai_manager.py` - CivitAI integration

**Frontend Files**:
- `frontend/components/model_lora_management.py` - CivitAI download UI

**Key Functions**:
- `download_lora_model()` - Download from CivitAI
- `get_updated_lora_files()` - Refresh LoRA list

---

### API Server
**Description**: SD WebUI-style API for external integrations  
**Backend Files**:
- `backend/api_server.py` - HTTP API server (`APIServer`)
- `backend/api_manager.py` - API endpoint management

**API Endpoints**:
- `POST /sdapi/v1/txt2img` - Text-to-image generation
- `POST /sdapi/v1/img2img` - Image-to-image generation
- `POST /sdapi/v1/controlnet` - ControlNet generation
- `POST /api/upscale` - Image upscaling

**Launch**:
- `python -m backend.api_server [host] [port]`
- Default: `0.0.0.0:7861`

---

## Utility & Support Features

### Prompts Manager
**Description**: System prompt management and LLM integration  
**Backend Files**:
- `backend/prompts_manager.py` - Prompt file management

**Key Functions**:
- `get_prompt_files()` - List prompt files for tab
- `load_prompt_file()` - Load prompt file content
- `save_prompt_file()` - Save prompt file
- `read_system_prompt()` - Read system prompts
- `save_ollama_settings()` - Save Ollama configuration

---

### Output Viewer
**Description**: Browse all generated outputs with lazy loading, search, and metadata viewing
**Backend Files**:
- `backend/output_viewer_manager.py` - Output management, pagination, metadata extraction, caching

**Frontend Files**:
- `frontend/components/output_viewer.py` - Output viewer UI tab (`create_output_viewer_tab`)

**Key Classes**:
- `OutputViewerManager` - Manages output browsing, search, and thumbnails
- `OutputItem` - Data class for individual output items

**Key Functions**:
- `get_outputs_paginated()` - Get paginated list of outputs
- `get_outputs_for_gallery()` - Get outputs formatted for Gradio Gallery
- `get_output_metadata()` - Extract metadata from output file
- `search_outputs()` - Search outputs by prompt text
- `delete_output()` - Delete an output file
- `get_settings_json()` - Get generation settings as JSON for copying

**Features**:
- Seamless infinite scroll with pre-fetching (loads at 70% scroll)
- Memory-efficient windowed virtualization (60 images max in memory)
- Search by prompt text
- View full metadata (prompt, model, seed, steps, guidance, LoRAs, timestamp)
- Copy generation settings to clipboard
- Delete outputs with confirmation
- Automatic thumbnail generation for performance

---

### Post Processing
**Description**: Image post-processing utilities
**Backend Files**:
- `backend/post_processing.py` - Post-processing functions

**Key Functions**:
- `update_dimensions_on_image_change()` - Adjust dimensions
- `update_dimensions_on_scale_change()` - Scale calculations

---

### MLX Utilities
**Description**: MLX memory management and optimization  
**Backend Files**:
- `backend/mlx_utils.py` - MLX utilities

**Key Functions**:
- `force_mlx_cleanup()` - Clean up MLX memory
- `print_memory_usage()` - Monitor memory usage

---

### MFLUX Compatibility
**Description**: Compatibility layer for different MFLUX versions  
**Backend Files**:
- `backend/mflux_compat.py` - Version compatibility shims

**Key Classes**:
- `Config` - Configuration compatibility
- `ModelConfig` - Model configuration compatibility
- `RuntimeConfig` - Runtime configuration compatibility

---

## Frontend Components

### Main UI
**Description**: Main Gradio interface assembly  
**Frontend Files**:
- `frontend/gradioui.py` - Main UI orchestration

**Key Functions**:
- `create_gradio_interface()` - Build main interface
- Imports and assembles all feature tabs

---

### Component Index
All frontend components follow the pattern: `create_[feature]_tab()`

| Component | File | Function |
|-----------|------|----------|
| Easy MFLUX | `easy_mflux.py` | `create_easy_mflux_tab()` |
| Advanced Generate | `advanced_generate.py` | `create_advanced_generate_tab()` |
| ControlNet | `controlnet.py` | `create_controlnet_tab()` |
| Image-to-Image | `image_to_image.py` | `create_image_to_image_tab()` |
| Fill | `fill.py` | `create_fill_tab()` |
| Depth | `depth.py` | `create_depth_tab()` |
| Redux | `redux.py` | `create_redux_tab()` |
| Upscale | `upscale.py` | `create_upscale_tab()` |
| CatVTON | `catvton.py` | `create_catvton_tab()` |
| IC-Edit | `ic_edit.py` | `create_ic_edit_tab()` |
| Concept Attention | `concept_attention.py` | `create_concept_attention_tab()` |
| Kontext | `kontext.py` | `create_kontext_tab()` |
| Canvas | `canvas.py` | `create_canvas_tab()` |
| Qwen Image | `qwen_image.py` | `create_qwen_image_tab()` |
| Qwen Edit | `qwen_edit.py` | `create_qwen_edit_tab()` |
| FIBO | `fibo.py` | `create_fibo_tab()` |
| Z-Image Turbo | `z_image_turbo.py` | `create_z_image_turbo_tab()` |
| Flux2 Generate | `flux2_generate.py` | `create_flux2_generate_tab()` |
| Flux2 Edit | `flux2_edit.py` | `create_flux2_edit_tab()` |
| In-Context LoRA | `in_context_lora.py` | `create_in_context_lora_tab()` |
| Auto Seeds | `auto_seeds.py` | `create_auto_seeds_tab()` |
| Dynamic Prompts | `dynamic_prompts.py` | `create_dynamic_prompts_tab()` |
| Config Manager | `config_manager.py` | `create_config_tab()` |
| Model/LoRA Mgmt | `model_lora_management.py` | `create_model_lora_management_tab()` |
| Dreambooth Training | `dreambooth_fine_tuning.py` | `create_dreambooth_fine_tuning_tab()` |
| LLM Settings | `llmsettings.py` | `create_llm_settings()` |
| Output Viewer | `output_viewer.py` | `create_output_viewer_tab()` |

---

## Prompt Management

### Prompt Directory Structure
```
prompts/
├── dynamic_prompts_config.json     # Dynamic prompts configuration
├── canvas/
│   └── canvas_generation_prompt.md # Canvas prompt enhancement
├── catvton/
│   └── virtual_tryon_prompt.md     # Virtual try-on prompts
├── concept_attention/
│   └── concept_focus_prompt.md     # Concept attention prompts
├── controlnet/
│   └── controlnet_prompt.md        # ControlNet prompts
├── depth/
│   └── depth_guided_prompt.md      # Depth-guided prompts
├── easy-advanced-prompt/
│   ├── cartoon.md                  # Cartoon style preset
│   ├── digitalart.md               # Digital art preset
│   ├── enhance.md                  # Enhancement preset
│   └── realistic.md                # Realistic style preset
├── fill/
│   ├── inpaint_prompt.md           # Inpainting prompts
│   └── outpaint_prompt.md          # Outpainting prompts
├── ic_edit/
│   ├── full_prompt_editing_prompt.md    # Full editing prompts
│   └── instruction_editing_prompt.md    # Instruction-based editing
├── image-to-image/
│   └── image_to_image_prompt.md    # img2img prompts
├── kontext/
│   └── kontext_editing_prompt.md   # Kontext editing prompts
├── redux/
│   └── redux_transformation_prompt.md   # Redux variation prompts
└── upscale/
    └── (upscale prompts)
```

---

## Quick Reference: Common Tasks

### Working on Text-to-Image Generation
**Files to Read**:
1. `backend/flux_manager.py` - Core generation logic
2. `frontend/components/easy_mflux.py` or `advanced_generate.py` - UI
3. `backend/generation_workflow.py` - Workflow integration

### Working on LoRA Support
**Files to Read**:
1. `backend/lora_manager.py` - LoRA management
2. `backend/lora_converter.py` - Format conversion
3. `frontend/components/model_lora_management.py` - UI

### Working on Model Management
**Files to Read**:
1. `backend/model_manager.py` - Model management
2. `backend/huggingface_manager.py` - HF integration
3. `backend/export_manager.py` - Quantization/export

### Working on Prompt Enhancement
**Files to Read**:
1. `backend/prompts_manager.py` - Prompt file management
2. `backend/ollama_manager.py` - Ollama integration
3. `backend/mlx_vlm_manager.py` - VLM integration
4. `frontend/components/llmsettings.py` - LLM settings UI

### Working on Dynamic Features
**Files to Read**:
1. `backend/dynamic_prompts_manager.py` - Dynamic prompts
2. `backend/auto_seeds_manager.py` - Auto seeds
3. `backend/generation_workflow.py` - Workflow orchestration

### Working on Image Editing Features
**Files to Read**:
1. `backend/fill_manager.py` - Inpainting/outpainting
2. `backend/ic_edit_manager.py` - In-context editing
3. `backend/kontext_manager.py` - Kontext editing
4. `backend/qwen_manager.py` - Qwen editing

### Working on Specialized Models
**Files to Read**:
1. `backend/flux2_manager.py` - Flux2 Klein
2. `backend/qwen_manager.py` - Qwen models
3. `backend/fibo_manager.py` - FIBO
4. `backend/z_image_manager.py` - Z-Image Turbo

### Working on Training
**Files to Read**:
1. `backend/training_manager.py` - Training orchestration
2. `backend/custom_train.py` - Training logic
3. `backend/custom_trainer.py` - Trainer implementation
4. `frontend/components/dreambooth_fine_tuning.py` - UI

### Working on API
**Files to Read**:
1. `backend/api_server.py` - HTTP API server
2. `backend/api_manager.py` - Endpoint management

### Working on Output Viewer
**Files to Read**:
1. `backend/output_viewer_manager.py` - Output browsing, pagination, metadata
2. `frontend/components/output_viewer.py` - Gallery UI and lazy loading
3. `backend/metadata_config_manager.py` - Metadata extraction utilities

---

## Architecture Overview

### Request Flow
```
User Input (Frontend)
    ↓
Gradio UI Component (frontend/components/*.py)
    ↓
Manager Function (backend/*_manager.py)
    ↓
Generation Workflow (backend/generation_workflow.py)
    ↓
MFLUX Model (Flux1, Flux2, Qwen, etc.)
    ↓
Output (images saved to output/)
```

### Feature Integration Flow
```
Generation Request
    ↓
Config Manager → Load/Validate Config
    ↓
Dynamic Prompts Manager → Process Wildcards
    ↓
Auto Seeds Manager → Select Seed
    ↓
LoRA Manager → Load LoRAs
    ↓
Model Manager → Load Model
    ↓
Generate Image
    ↓
Stepwise Output → Save Progress
    ↓
Metadata Manager → Save Metadata
    ↓
Return Result
```

---

## Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MFLUX_SERVER_NAME` | WebUI server address | `127.0.0.1` |
| `MFLUX_SERVER_PORT` | WebUI server port | `7860` |
| `LORA_LIBRARY_PATH` | External LoRA paths | `lora` |
| `MFLUX_UPSCALE_STEPS` | Upscale steps | `12` |
| `MFLUX_UPSCALE_STRENGTH` | Upscale strength | `0.75` |
| `MFLUX_UPSCALE_QUANTIZE` | Upscale quantization | None |
| `MLX_VLM_CAPTION_MODEL` | Caption model override | Florence-2-large-ft-bf16 |
| `HUGGINGFACE_HUB_TOKEN` | HF API token | - |

---

## File Organization Summary

```
MFLUX-WEBUI/
├── webui.py                    # Main entry point
├── requirements.txt            # Dependencies
├── readme.md                   # Project documentation
├── backend/                    # Backend logic
│   ├── *_manager.py           # Feature managers
│   ├── flux_manager.py        # Core Flux generation
│   ├── generation_workflow.py # Workflow orchestration
│   ├── model_manager.py       # Model management
│   ├── lora_manager.py        # LoRA management
│   ├── output_viewer_manager.py # Output browsing & search
│   └── ...
├── frontend/                   # UI components
│   ├── gradioui.py            # Main UI
│   └── components/            # Feature tabs
│       ├── easy_mflux.py
│       ├── advanced_generate.py
│       ├── output_viewer.py   # Output viewer with lazy loading
│       └── ...
├── prompts/                    # Prompt templates
│   ├── dynamic_prompts_config.json
│   └── [feature]/             # Feature-specific prompts
├── models/                     # Downloaded models
├── lora/                       # LoRA files
├── output/                     # Generated images
│   └── .thumbnails/           # Cached thumbnails for viewer
└── configs/                    # Configuration files
```

---

## Version Information

**Document Version**: 1.1
**MFLUX WebUI Version**: 0.15.4
**Last Updated**: 2026-02-02

### Changelog
- v1.1: Added Output Viewer feature documentation

---

## Notes for AI Agents

1. **Feature Location**: Use the feature name to locate the corresponding `*_manager.py` file in `backend/`
2. **UI Components**: Frontend components are in `frontend/components/` with consistent naming
3. **Workflow Integration**: Most features integrate with `backend/generation_workflow.py`
4. **Configuration**: Config files are in `configs/`, prompts in `prompts/`
5. **Model Support**: Check `backend/model_manager.py` for model compatibility
6. **API Integration**: Use `backend/api_server.py` for external API access

### Common Patterns
- Generation functions: `generate_[feature]_gradio()`
- UI creation: `create_[feature]_tab()`
- Model initialization: `get_or_create_flux_[feature]()`
- Manager classes: `[Feature]Manager`

### Testing a Feature
1. Read the `*_manager.py` file for backend logic
2. Check the corresponding `frontend/components/*.py` for UI
3. Review `prompts/[feature]/` for prompt templates
4. Check `backend/generation_workflow.py` for integration

---

**End of Features Index**
