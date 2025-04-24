"""
node_parameter_mapping.py
Description: Node parameter mapping utilities for direct extraction of metadata from workflows.
    Maps node_type.parameter_name to metadata paths for structured extraction.
Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
Version: 1.0.0
Date: [March 2025]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.

Dependencies:
This code depends on several third-party libraries, each with its own license:

"""

from typing import Dict, Any, List, Optional, Tuple, Set
import datetime

# Define mapping from node type + parameter name to metadata path
NODE_PARAMETER_MAPPING = {
    # Base Models
    "CheckpointLoader.ckpt_name": ["ai_info", "generation", "base_model", "name"],
    "CheckpointLoaderSimple.ckpt_name": ["ai_info", "generation", "base_model", "name"],
    "DiffusersLoader.model_path": ["ai_info", "generation", "base_model", "name"],
    "UNETLoader.unet_name": ["ai_info", "generation", "base_model", "unet"],
    "SDXLCheckpointLoader.ckpt_name": ["ai_info", "generation", "base_model", "name"],
    "BaseModelLoader.ckpt_name": ["ai_info", "generation", "base_model", "name"],
    "ModelLoader.ckpt_name": ["ai_info", "generation", "base_model", "name"],
    "ModelSelector.ckpt_name": ["ai_info", "generation", "base_model", "name"],

    # Sampling Parameters
    "KSampler.sampler_name": ["ai_info", "generation", "sampling", "sampler"],
    "KSampler.scheduler": ["ai_info", "generation", "sampling", "scheduler"],
    "KSampler.steps": ["ai_info", "generation", "sampling", "steps"],
    "KSampler.cfg": ["ai_info", "generation", "sampling", "cfg_scale"],
    "KSampler.cfg_scale": ["ai_info", "generation", "sampling", "cfg_scale"],
    "KSampler.guidance_scale": ["ai_info", "generation", "sampling", "cfg_scale"],
    "KSampler.seed": ["ai_info", "generation", "sampling", "seed"],
    "KSampler.seed_random": ["ai_info", "generation", "sampling", "seed"],
    "KSampler.denoise": ["ai_info", "generation", "sampling", "denoise"],
    
    # Advanced Samplers
    "KSamplerAdvanced.sampler_name": ["ai_info", "generation", "sampling", "sampler"],
    "KSamplerAdvanced.scheduler": ["ai_info", "generation", "sampling", "scheduler"],
    "KSamplerAdvanced.steps": ["ai_info", "generation", "sampling", "steps"],
    "KSamplerAdvanced.cfg": ["ai_info", "generation", "sampling", "cfg_scale"],
    "KSamplerAdvanced.seed": ["ai_info", "generation", "sampling", "seed"],
    "KSamplerAdvanced.denoise": ["ai_info", "generation", "sampling", "denoise"],
    
    # Custom Samplers
    "SamplerCustom.sampler_name": ["ai_info", "generation", "sampling", "sampler"],
    "SamplerCustom.scheduler": ["ai_info", "generation", "sampling", "scheduler"],
    "SamplerCustom.steps": ["ai_info", "generation", "sampling", "steps"],
    "SamplerCustom.cfg": ["ai_info", "generation", "sampling", "cfg_scale"],
    "SamplerCustom.seed": ["ai_info", "generation", "sampling", "seed"],
    
    # Sampler Selectors
    "KSamplerSelect.sampler_name": ["ai_info", "generation", "sampling", "sampler"],
    "KSamplerSelect.scheduler": ["ai_info", "generation", "sampling", "scheduler"],
    "SamplerSelector.sampler_name": ["ai_info", "generation", "sampling", "sampler"],
    "SchedulerSelector.scheduler_name": ["ai_info", "generation", "sampling", "scheduler"],
    
    # Basic Scheduler
    "BasicScheduler.scheduler": ["ai_info", "generation", "sampling", "scheduler"],
    "BasicScheduler.steps": ["ai_info", "generation", "sampling", "steps"],
    "BasicScheduler.denoise": ["ai_info", "generation", "sampling", "denoise"],
    
    # Random Noise
    "RandomNoise.noise_seed": ["ai_info", "generation", "sampling", "seed"],
    
    # Flux Parameters
    "ModelSamplingFlux.max_shift": ["ai_info", "generation", "flux", "max_shift"],
    "ModelSamplingFlux.base_shift": ["ai_info", "generation", "flux", "base_shift"],
    "ModelSamplingFlux.flux_frac": ["ai_info", "generation", "flux", "fraction"],
    "ModelSamplingFlux.flux_mode": ["ai_info", "generation", "flux", "mode"],
    "FluxModelConverter.flux_max_shift": ["ai_info", "generation", "flux", "max_shift"],
    "FluxModelConverter.flux_base_shift": ["ai_info", "generation", "flux", "base_shift"],
    "FluxGuidance.guidance": ["ai_info", "generation", "flux", "guidance"],
    
    # CLIP Models
    "CLIPLoader.clip_name": ["ai_info", "generation", "modules", "clip", "name"],
    "CLIPVisionLoader.clip_name": ["ai_info", "generation", "modules", "clip_vision", "name"],
    "DualCLIPLoader.clip_name1": ["ai_info", "generation", "modules", "clip", "primary"],
    "DualCLIPLoader.clip_name2": ["ai_info", "generation", "modules", "clip", "secondary"],
    "TripleCLIPLoader.clip_name1": ["ai_info", "generation", "modules", "clip", "primary"],
    "TripleCLIPLoader.clip_name2": ["ai_info", "generation", "modules", "clip", "secondary"],
    "TripleCLIPLoader.clip_name3": ["ai_info", "generation", "modules", "clip", "tertiary"],
    
    # VAE Models
    "VAELoader.vae_name": ["ai_info", "generation", "modules", "vae", "name"],
    "VAEDecoderLoad.vae_name": ["ai_info", "generation", "modules", "vae", "name"],
    
    # LoRA Models
    "LoraLoader.lora_name": ["ai_info", "generation", "modules", "loras", "name"],
    "LoraLoader.strength_model": ["ai_info", "generation", "modules", "loras", "strength_model"],
    "LoraLoader.strength_clip": ["ai_info", "generation", "modules", "loras", "strength_clip"],
    "LoraLoaderModelOnly.lora_name": ["ai_info", "generation", "modules", "loras", "name"],
    "LoraLoaderModelOnly.strength_model": ["ai_info", "generation", "modules", "loras", "strength_model"],
    "LoraLoaderTagsQuery.lora_name": ["ai_info", "generation", "modules", "loras", "name"],
    "LoraLoaderTagsQuery.strength_model": ["ai_info", "generation", "modules", "loras", "strength_model"],
    "LoraLoaderTagsQuery.strength_clip": ["ai_info", "generation", "modules", "loras", "strength_clip"],
    
    # ControlNet Models
    "ControlNetLoader.control_net_name": ["ai_info", "generation", "modules", "controlnets", "name"],
    "ControlNetApply.strength": ["ai_info", "generation", "modules", "controlnets", "strength"],
    
    # IP Adapter Models
    "IPAdapterModelLoader.ipadapter_file": ["ai_info", "generation", "modules", "ip_adapters", "name"],
    
    # Style Models
    "StyleModelLoader.style_model_name": ["ai_info", "generation", "modules", "style_models", "name"],
    "StyleModelApplyAdvanced.strength": ["ai_info", "generation", "modules", "style_models", "strength"],
    
    # Upscale Models
    "UpscaleModelLoader.model_name": ["ai_info", "generation", "modules", "upscalers", "name"],
    
    # Colorization
    "DDColor_Colorize.model_input_size": ["ai_info", "generation", "colorization", "input_size"],
    "DDColor_Colorize.checkpoint": ["ai_info", "generation", "colorization", "model"],
    
    # Conditioning
    "CLIPTextEncode.text": ["ai_info", "generation", "prompt"],
    "CLIPTextEncodeSD3.text": ["ai_info", "generation", "prompt"],
    "SD3NegativeConditioning+.end": ["ai_info", "generation", "negative_conditioning", "end"],

    # Image Processing
    "ImageResize+.width": ["ai_info", "generation", "image_processing", "resize", "width"],
    "ImageResize+.height": ["ai_info", "generation", "image_processing", "resize", "height"],
    "ImageResize+.interpolation": ["ai_info", "generation", "image_processing", "resize", "interpolation"],
    "ImageResize+.method": ["ai_info", "generation", "image_processing", "resize", "method"],
    "ImageScaleToTotalPixels.upscale_method": ["ai_info", "generation", "image_processing", "scale", "method"],
    "ImageScaleToTotalPixels.megapixels": ["ai_info", "generation", "image_processing", "scale", "megapixels"],

    # Tonemapping
    "CV2Tonemap.gamma": ["ai_info", "generation", "image_processing", "tonemap", "gamma"],
    "CV2Tonemap.mult": ["ai_info", "generation", "image_processing", "tonemap", "multiplier"],
    "CV2TonemapDrago.gamma": ["ai_info", "generation", "image_processing", "tonemap_drago", "gamma"],
    "CV2TonemapDrago.saturation": ["ai_info", "generation", "image_processing", "tonemap_drago", "saturation"],
    "CV2TonemapDrago.bias": ["ai_info", "generation", "image_processing", "tonemap_drago", "bias"],
    "CV2TonemapDrago.mult": ["ai_info", "generation", "image_processing", "tonemap_drago", "multiplier"],

    # AI Analysis
    "JoyCaption.prompt": ["ai_info", "generation", "ai_analysis", "caption", "prompt"],
    "JoyCaption.model": ["ai_info", "generation", "ai_analysis", "caption", "model"],
    "JoyCaption.max_new_tokens": ["ai_info", "generation", "ai_analysis", "caption", "max_tokens"],
    "JoyCaption.temperature": ["ai_info", "generation", "ai_analysis", "caption", "temperature"],
    "ComfyUIPixtralVision.prompt": ["ai_info", "generation", "ai_analysis", "vision", "prompt"],
    "ComfyUIPixtralVision.temperature": ["ai_info", "generation", "ai_analysis", "vision", "temperature"],
    "ComfyUIPixtralVision.maximum_tokens": ["ai_info", "generation", "ai_analysis", "vision", "max_tokens"],
    "Joytag.tag_number": ["ai_info", "generation", "ai_analysis", "tagging", "count"],

    # Text Inputs
    "Text Multiline.text": ["ai_info", "generation", "text_inputs", "multiline"],
    "Simple String.string": ["ai_info", "generation", "text_inputs", "simple"],
    "Text Concatenate.delimiter": ["ai_info", "generation", "text_inputs", "concatenate", "delimiter"],
    "Text Concatenate.clean_whitespace": ["ai_info", "generation", "text_inputs", "concatenate", "clean_whitespace"],

    # Dimensions
    "EmptyLatentImage.width": ["ai_info", "generation", "dimensions", "width"],
    "EmptyLatentImage.height": ["ai_info", "generation", "dimensions", "height"],
    "EmptyLatentImage.batch_size": ["ai_info", "generation", "dimensions", "batch_size"],
    "EmptySD3LatentImage.width": ["ai_info", "generation", "dimensions", "width"],
    "EmptySD3LatentImage.height": ["ai_info", "generation", "dimensions", "height"],
    "EmptySD3LatentImage.batch_size": ["ai_info", "generation", "dimensions", "batch_size"],
    "SDXLEmptyLatentSizePicker+.width_override": ["ai_info", "generation", "dimensions", "width"],
    "SDXLEmptyLatentSizePicker+.height_override": ["ai_info", "generation", "dimensions", "height"],
    "SDXLEmptyLatentSizePicker+.batch_size": ["ai_info", "generation", "dimensions", "batch_size"],
    
    # Reference images
    "LoadImage.image": ["ai_info", "generation", "reference", "image"]
}

# Parameters that should be treated as negative prompts if found
NEGATIVE_PROMPT_INDICATORS = [
    "negative",
    "neg"
]

def extract_by_parameter_mapping(nodes: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """
    Extract metadata using direct node parameter mapping
    
    Args:
        nodes: Dictionary of workflow nodes
        debug: Enable debug output
        
    Returns:
        dict: Structured metadata dictionary
    """
    metadata = {}
    lora_nodes = {}  # Track LoRA nodes to group their parameters
    controlnet_nodes = {}  # Track ControlNet nodes to group their parameters
    
    # Discover and log actual node types and parameters
    if debug:
        discovered = discover_node_parameters(nodes)
        print(f"[NodeParameterMapping] Discovered node types and parameters:")
        for node_type, params in discovered.items():
            print(f"  {node_type}: {', '.join(params)}")
    
    # Process all nodes
    for node_id, node in nodes.items():
        node_type = node.get('class_type', '')
        inputs = node.get('inputs', {})
        
        # Skip if no inputs or no node type
        if not inputs or not node_type:
            continue
        
        # Check if this is a negative prompt node
        is_negative_prompt = False
        if node_type == "CLIPTextEncode" and '_meta' in node and 'title' in node['_meta']:
            title = node['_meta']['title'].lower()
            is_negative_prompt = any(indicator in title for indicator in NEGATIVE_PROMPT_INDICATORS)
            
        # Process all parameters
        for param_name, param_value in inputs.items():
            # Create the lookup key
            lookup_key = f"{node_type}.{param_name}"
            
            # Skip if not in our mapping
            if lookup_key not in NODE_PARAMETER_MAPPING:
                if debug:
                    print(f"[NodeParameterMapping] No mapping for: {lookup_key}")
                continue
                
            # Try to resolve references
            if isinstance(param_value, list) and len(param_value) == 2:
                # Resolve the reference
                resolved_value = resolve_reference(param_value, nodes)
                if resolved_value == param_value:  # If resolution failed
                    if debug:
                        print(f"[NodeParameterMapping] Couldn't resolve reference: {lookup_key} = {param_value}")
                    continue
                param_value = resolved_value
                
            # Get metadata path
            path = NODE_PARAMETER_MAPPING[lookup_key]
            
            # Special handling for prompt
            if lookup_key == "CLIPTextEncode.text":
                if is_negative_prompt:
                    # Override path for negative prompt
                    path = ["ai_info", "generation", "negative_prompt"]
            
            # Special handling for LoRAs
            if path[:-1] == ["ai_info", "generation", "loras"]:
                # Group LoRA parameters by node ID
                if node_id not in lora_nodes:
                    lora_nodes[node_id] = {
                        "name": None,
                        "strength_model": 1.0,
                        "strength_clip": 1.0
                    }
                
                # Store parameter value
                lora_nodes[node_id][path[-1]] = param_value
                continue
                
            # Special handling for ControlNets
            if path[:-1] == ["ai_info", "generation", "control_nets"]:
                # Group ControlNet parameters by node ID
                if node_id not in controlnet_nodes:
                    controlnet_nodes[node_id] = {
                        "name": None,
                        "strength": 1.0
                    }
                
                # Store parameter value
                controlnet_nodes[node_id][path[-1]] = param_value
                continue
            
            # Regular parameter - set in metadata
            current = metadata
            
            # Create path
            for i, path_part in enumerate(path[:-1]):
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
            
            # Set value at path
            current[path[-1]] = param_value
    
    # Process LoRAs after all nodes
    if lora_nodes:
        loras = []
        
        # Convert to list format
        for node_id, lora_info in lora_nodes.items():
            if lora_info["name"]:  # Only add if name was found
                loras.append({
                    "name": lora_info["name"],
                    "strength_model": lora_info["strength_model"],
                    "strength_clip": lora_info["strength_clip"]
                })
        
        # Add to metadata if any valid LoRAs found
        if loras:
            # Create path if needed
            if "ai_info" not in metadata:
                metadata["ai_info"] = {}
            if "generation" not in metadata["ai_info"]:
                metadata["ai_info"]["generation"] = {}
            
            metadata["ai_info"]["generation"]["loras"] = loras
            
    # Process ControlNets after all nodes
    if controlnet_nodes:
        controlnets = []
        
        # Convert to list format
        for node_id, controlnet_info in controlnet_nodes.items():
            if controlnet_info["name"]:  # Only add if name was found
                controlnets.append({
                    "name": controlnet_info["name"],
                    "strength": controlnet_info["strength"]
                })
        
        # Add to metadata if any valid ControlNets found
        if controlnets:
            # Create path if needed
            if "ai_info" not in metadata:
                metadata["ai_info"] = {}
            if "generation" not in metadata["ai_info"]:
                metadata["ai_info"]["generation"] = {}
            
            metadata["ai_info"]["generation"]["controlnets"] = controlnets
    
    # Add timestamp
    if "ai_info" in metadata and "generation" in metadata["ai_info"]:
        metadata["ai_info"]["generation"]["timestamp"] = datetime.datetime.now().isoformat()
            
    # Add debug output at end
    if debug:
        print(f"[NodeParameterMapping] Extracted metadata: {metadata}")
        if "ai_info" in metadata and "generation" in metadata["ai_info"]:
            print(f"[NodeParameterMapping] Generation parameters extracted:")
            for key, value in metadata["ai_info"]["generation"].items():
                print(f"  - {key}: {value}")
    
    return metadata

def resolve_reference(ref_value, nodes):
    """
    Resolve a node reference to its actual value
    
    Args:
        ref_value: Reference value [node_id, output_index]
        nodes: Dictionary of all nodes
        
    Returns:
        Any: Resolved value if possible, or the reference
    """
    if not isinstance(ref_value, list) or len(ref_value) != 2:
        return ref_value
        
    node_id, output_idx = ref_value
    
    # Check if referenced node exists
    if str(node_id) not in nodes:
        return ref_value
        
    ref_node = nodes[str(node_id)]
    node_type = ref_node.get('class_type', '')
    
    # Handle common value-type nodes
    if node_type == 'PrimitiveNode' and 'value' in ref_node.get('inputs', {}):
        return ref_node['inputs']['value']
        
    if node_type == 'StringNode' and 'string' in ref_node.get('inputs', {}):
        return ref_node['inputs']['string']
        
    # Handle specific nodes we recognize
    if node_type == 'SamplerNode' and 'sampler_name' in ref_node.get('inputs', {}):
        return ref_node['inputs']['sampler_name']
        
    if node_type == 'SchedulerNode' and 'scheduler' in ref_node.get('inputs', {}):
        return ref_node['inputs']['scheduler']
    
    # Return the node type as fallback
    return node_type

def discover_node_parameters(nodes: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Discover actual node types and their parameters in a workflow
    
    Args:
        nodes: Dictionary of workflow nodes
        
    Returns:
        dict: Mapping of node types to parameter names
    """
    discovered = {}
    
    for node_id, node in nodes.items():
        node_type = node.get('class_type', '')
        inputs = node.get('inputs', {})
        
        if not node_type or not inputs:
            continue
            
        if node_type not in discovered:
            discovered[node_type] = []
            
        for param_name in inputs.keys():
            if param_name not in discovered[node_type]:
                discovered[node_type].append(param_name)
    
    return discovered