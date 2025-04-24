"""
ComfyUI Node: eric_metadata_save_image_v097
Description: Saves images with embedded metadata and supports multiple formats including layers.
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
- torch: BSD 3-Clause
- numpy: BSD 3-Clause
- opencv-python (cv2): Apache 2.0
- psd-tools: MIT License
- pillow: PIL Software License (Python Imaging Library) 

"""

import json
import datetime
import numpy as np
import torch
import traceback
from PIL import Image, ImageCms, TiffImagePlugin, TiffTags, ImageChops
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import Resource, BlendMode
from psd_tools.psd.image_resources import ImageResource, ImageResources
import folder_paths
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import re
import cv2
import io
import os

# Try to import pypng for better 16-bit support
try:
    import png
    HAS_PYPNG = True
except ImportError:
    HAS_PYPNG = False
    print("[MetadataSaveImage] pypng library not available - 16-bit PNG will fall back to OpenCV")

try:
    # Try importing vtracer for SVG conversion
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False
    print("[MetadataSaveImage] vtracer not available - SVG export will be disabled")

from Metadata_system import MetadataService
from Metadata_system.eric_metadata.utils.workflow_parser import WorkflowParser
from Metadata_system.eric_metadata.utils.workflow_extractor import WorkflowExtractor
from Metadata_system.eric_metadata.utils.workflow_metadata_processor import WorkflowMetadataProcessor


# Initialize color profiles - will be populated on first use
ICC_PROFILES = {
    'sRGB v4 Appearance': None,  # Default, modern sRGB profile
    'sRGB v4 Preference': None,
    'sRGB v4 Display Class': None,
    'Adobe RGB': None,
    'ProPhoto RGB': None,
    'None': None
}

# Profile filenames mapping
PROFILE_FILENAMES = {
    'sRGB v4 Appearance': 'sRGB_ICC_v4_Appearance.icc',
    'sRGB v4 Preference': 'sRGB_v4_ICC_preference.icc',
    'sRGB v4 Display Class': 'sRGB_v4_ICC_preference_displayclass.icc',
    'Adobe RGB': 'AdobeRGB1998.icc',
    'ProPhoto RGB': 'ProPhoto.icm',
}

# Color space conversion intents
COLOR_INTENTS = {
    'Perceptual': 0,  # Optimizes for natural-looking images
    'Relative Colorimetric': 1,  # Maintains color accuracy, maps white point
    'Saturation': 2,  # Preserves saturation, good for graphics  
    'Absolute Colorimetric': 3,  # Preserves exact colors, no white point adjustment
}

class MetadataAwareSaveImage:
    """SaveImage node that embeds metadata during save, with support for multiple formats including layers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Define available file formats based on installed packages
        file_formats = ["png", "jpg", "webp", "tiff", "apng"]  # Add APNG format
        if HAS_VTRACER:
            file_formats.append("svg")
            
        try:
            from PIL import PsdImagePlugin
            file_formats.append("psd")
        except ImportError:
            pass
        
        # Define blend modes for layers
        blend_modes = [
            "normal", "multiply", "screen", "overlay",
            "add", "subtract", "difference", "darken", "lighten",
            "color_dodge", "color_burn"
        ]
        
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Base name for saved files. Can include formatting like %date:yyyy-MM-dd%"}), 
                "file_format": (file_formats, {"default": "png", "tooltip": "Image format to save in"}), 
            },
            "optional": {
                # File naming options
                "include_project": ("BOOLEAN", {"default": False, "tooltip": "Add project name to filename"}), 
                "include_datetime": ("BOOLEAN", {"default": True, "tooltip": "Add date and time to filename"}), 
                
                # Layer-related inputs with opacity sliders
                "mask_layer_1": ("IMAGE", {"optional": True, "tooltip": "Optional mask layer 1"}), 
                "mask_1_name": ("STRING", {"default": "Mask 1", "tooltip": "Name for mask layer 1"}), 
                "mask_1_blend_mode": (blend_modes, {"default": "normal", "tooltip": "Blend mode for mask layer 1"}), 
                "mask_1_opacity": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Opacity for mask layer 1 (0-255)"}),
                
                "mask_layer_2": ("IMAGE", {"optional": True, "tooltip": "Optional mask layer 2"}), 
                "mask_2_name": ("STRING", {"default": "Mask 2", "tooltip": "Name for mask layer 2"}), 
                "mask_2_blend_mode": (blend_modes, {"default": "normal", "tooltip": "Blend mode for mask layer 2"}), 
                "mask_2_opacity": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Opacity for mask layer 2 (0-255)"}),
                
                "overlay_layer_1": ("IMAGE", {"optional": True, "tooltip": "Optional overlay layer 1"}), 
                "overlay_1_name": ("STRING", {"default": "Overlay 1", "tooltip": "Name for overlay layer 1"}), 
                "overlay_1_blend_mode": (blend_modes, {"default": "normal", "tooltip": "Blend mode for overlay layer 1"}), 
                "overlay_1_opacity": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Opacity for overlay layer 1 (0-255)"}),
                
                "overlay_layer_2": ("IMAGE", {"optional": True, "tooltip": "Optional overlay layer 2"}), 
                "overlay_2_name": ("STRING", {"default": "Overlay 2", "tooltip": "Name for overlay layer 2"}), 
                "overlay_2_blend_mode": (blend_modes, {"default": "normal", "tooltip": "Blend mode for overlay layer 2"}), 
                "overlay_2_opacity": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "Opacity for overlay layer 2 (0-255)"}),
                
                # Output location options
                "custom_output_directory": ("STRING", {
                    "default": "", 
                    "tooltip": "Override the default output location. For relative paths, don't start with backslash. Use 'images/subfolder' format. Leave empty to use default output folder."
                }), 
                "output_path_mode": (["Absolute Path", "Subfolder in Output"], 
                    {"default": "Subfolder in Output", 
                    "tooltip": "How to interpret the custom output directory: as an absolute path, as a subfolder in the standard output folder"}), 
                "filename_format": (["Default (padded zeros)", "Simple (file1.png)"], 
                {"default": "Default (padded zeros)", 
                    "tooltip": "How to format sequential filenames: with padded zeros or simple numbers"}), 
                
                # File format and quality options
                "quality_preset": (["Best Quality", "Balanced", "Smallest File"], 
                    {"default": "Best Quality", "tooltip": "Quality versus file size tradeoff"}), 
                "color_profile": (list(ICC_PROFILES.keys()), 
                    {"default": "sRGB v4 Appearance", "tooltip": "Color profile to embed in the image"}), 
                "rendering_intent": (list(COLOR_INTENTS.keys()), 
                    {"default": "Relative Colorimetric", "tooltip": "How colors are mapped when converting between color spaces"}), 
                "bit_depth": (["8-bit", "16-bit"], 
                    {"default": "8-bit", "tooltip": "Save in 8-bit (standard) or 16-bit (high precision) format. 16-bit only works with PNG."}), 
                
                # NEW: Alpha channel handling
                "alpha_mode": (["auto", "premultiplied", "straight", "matte"], 
                    {"default": "auto", "tooltip": "How to handle alpha channel. Auto=choose best for format, Matte=composite onto background color"}), 
                "matte_color": ("STRING", 
                    {"default": "#FFFFFF", "tooltip": "Background color when saving with matte. Use color name or hex value."}), 
                "save_alpha_separately": ("BOOLEAN", 
                    {"default": False, "tooltip": "Save alpha channel as separate image for formats that don't support transparency"}), 
                
                # NEW: APNG specific options
                "apng_fps": ("INT", 
                    {"default": 10, "min": 1, "max": 60, "tooltip": "Frames per second for animated PNG"}), 
                "apng_loops": ("INT", 
                    {"default": 0, "min": 0, "max": 99999, "tooltip": "Number of animation loops (0 = infinite)"}), 
                
                # Add secondary format options
                "additional_format": (["None"] + file_formats, {
                    "default": "None", 
                    "tooltip": "Save a second copy in this format (e.g., workflow in PNG plus JPG for sharing)"
                }), 
                "additional_format_quality": (["Best Quality", "Balanced", "Smallest File"], {
                    "default": "Balanced",
                    "tooltip": "Quality setting for the additional format"
                }), 
                "additional_format_suffix": ("STRING", {
                    "default": "_web", 
                    "tooltip": "Suffix for the additional format filename (e.g., image_web.jpg)"
                }), 
                "additional_format_embed_workflow": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "Whether to embed workflow data in the additional format (PNG only)"
                }), 
                "additional_format_color_profile": (list(ICC_PROFILES.keys()), {
                    "default": "sRGB v4 Appearance",
                    "tooltip": "Color profile for the additional format"
                }), 

                # SVG specific options
                "svg_colormode": (["color", "binary"], 
                    {"default": "color", "tooltip": "SVG color mode (only applies when saving as SVG)"}), 
                "svg_hierarchical": (["stacked", "cutout"], 
                    {"default": "stacked", "tooltip": "SVG hierarchy mode (only applies when saving as SVG)"}), 
                "svg_mode": (["spline", "polygon", "none"], 
                    {"default": "spline", "tooltip": "SVG curve mode (only applies when saving as SVG)"}), 
                
                # Metadata content options
                "enable_metadata": ("BOOLEAN", 
                    {"default": True, "tooltip": "Master switch to enable/disable all metadata writing. When disabled, no structured metadata will be written anywhere."}), 
                "title": ("STRING", 
                    {"default": "", "tooltip": "Image title"}), 
                "project": ("STRING", 
                    {"default": "", "tooltip": "Project name for organization"}), 
                "description": ("STRING", 
                    {"default": "", "tooltip": "Image description/caption"}), 
                "creator": ("STRING", 
                    {"default": "", "tooltip": "Creator/artist name"}), 
                "copyright": ("STRING", 
                    {"default": "", "tooltip": "Copyright information"}), 
                "keywords": ("STRING", 
                    {"default": "", "tooltip": "Comma-separated keywords/tags"}), 
                "custom_metadata": ("STRING", {
                    "default": "{}", 
                    "multiline": False, 
                    "tooltip": "Custom metadata in JSON format. Example: {\"basic\":{\"source\":\"My Collection\"},\"ai_info\":{\"generation\":{\"negative_prompt\":\"blurry\"}},\"custom_section\":{\"custom_field\":\"value\"}}"
                }), 
                
                # Metadata storage options
                "save_embedded": ("BOOLEAN", 
                    {"default": True, "tooltip": "Save structured metadata directly inside the image file (title, description, etc.) - distinct from workflow embedding."}), 
                "save_workflow_as_json": ("BOOLEAN", 
                    {"default": False, "tooltip": "Save workflow data as a separate JSON file"}), 
                "save_xmp": ("BOOLEAN", 
                    {"default": True, "tooltip": "Save metadata in XMP sidecar file (.xmp)"}), 
                "save_txt": ("BOOLEAN", 
                    {"default": True, "tooltip": "Save human-readable metadata in text file (.txt)"}), 
                "save_db": ("BOOLEAN", 
                    {"default": False, "tooltip": "Save metadata to database (if configured)"}), 
                    
                # Advanced options
                "save_individual_discovery": ("BOOLEAN", 
                    {"default": False, "tooltip": "Save individual discovery JSON and HTML files for each run (central discovery is always maintained)"}), 
                "debug_logging": ("BOOLEAN", 
                    {"default": False, "tooltip": "Enable detailed debug logging to console"}), 
                
                # Include workflow embedding option
                "embed_workflow": ("BOOLEAN", 
                    {"default": True, "tooltip": "Embed ComfyUI workflow graph data in the PNG file - enables re-loading workflow later (PNG only)"}), 
            },
            # Standard ComfyUI way to access workflow data
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filepath")
    FUNCTION = "save_with_metadata"
    CATEGORY = "Eric's Nodes/Output"
    OUTPUT_NODE = True
    # The following line is a direct copy from ComfyUI's SaveImage node
    OUTPUT_IS_LIST = (False, False)

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"  # Explicitly set to match ComfyUI's SaveImage
        self.prefix_append = ""
        
        # Set debug flag first
        self.debug = False  # Set to True for more detailed debugging info
        
        # Initialize metadata service and workflow parser
        self.metadata_service = MetadataService(debug=self.debug, human_readable_text=True)
        self.workflow_parser = WorkflowParser(debug=self.debug)
        
        # Add the workflow extractor
        self.workflow_extractor = WorkflowExtractor(debug=self.debug, discovery_mode=True)
        
        # Initialize the workflow metadata processor
        self.workflow_processor = WorkflowMetadataProcessor(debug=self.debug, discovery_mode=True)
    
        # Save node ID for debugging
        self.node_id = None
        
        # UUID for this instance
        import uuid
        self.instance_id = str(uuid.uuid4())
    
    def _save_as_tiff(self, image, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """Save image as TIFF with proper layers, color and 16-bit support"""
        try:
            # Convert to PIL image - ensuring we get the right format
            pil_image = self._convert_to_pil(image)
            print(f"[TIFF] Starting with image: {pil_image.mode}, {pil_image.size}")
            
            # Get dimensions
            width, height = pil_image.size
            
            # Get bit depth setting
            bit_depth = kwargs.get("bit_depth", "8-bit")
            is_16bit = bit_depth == "16-bit"
            
            # Get color profile
            color_profile_name = kwargs.get("color_profile", "sRGB v4 Appearance") 
            color_profile_data = self._get_color_profile(color_profile_name)
            
            # Prepare compression setting
            compression = kwargs.get("quality_preset", "Balanced")
            if compression == "Best Quality":
                tiff_compression = "lzw"
            elif compression == "Balanced":
                tiff_compression = "zip"
            else:  # "Smallest File"
                tiff_compression = "deflate"
                
            # Collect all layers (main image + overlays + masks)
            all_layers = []
            layer_names = []
            
            # Add main image
            all_layers.append(pil_image)
            layer_names.append("Main Image")
            
            # Add overlay layers
            for i in range(1, 3):  # Up to 2 overlays
                overlay_key = f"overlay_layer_{i}"
                name_key = f"overlay_{i}_name"
                
                if overlay_key in kwargs and kwargs[overlay_key] is not None:
                    try:
                        overlay_img = self._convert_to_pil(kwargs[overlay_key])
                        overlay_name = kwargs.get(name_key, f"Overlay {i}")
                        
                        all_layers.append(overlay_img)
                        layer_names.append(overlay_name)
                        print(f"[TIFF] Added overlay layer: {overlay_name}")
                    except Exception as e:
                        print(f"[TIFF] Error adding overlay {i}: {str(e)}")
            
            # Add mask layers
            for i in range(1, 3):  # Up to 2 masks
                mask_key = f"mask_layer_{i}"
                name_key = f"mask_{i}_name"
                
                if mask_key in kwargs and kwargs[mask_key] is not None:
                    try:
                        mask_img = self._convert_to_pil(kwargs[mask_key])
                        mask_name = kwargs.get(name_key, f"Mask {i}")
                        
                        # Convert to grayscale if needed
                        if mask_img.mode != 'L':
                            mask_img = mask_img.convert('L')
                            
                        all_layers.append(mask_img)
                        layer_names.append(mask_name)
                        print(f"[TIFF] Added mask layer: {mask_name}")
                    except Exception as e:
                        print(f"[TIFF] Error adding mask {i}: {str(e)}")
            
            # Handle 16-bit conversion if needed
            if is_16bit:
                # Convert all layers to 16-bit
                converted_layers = []
                for i, layer in enumerate(all_layers):
                    name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
                    
                    try:
                        if layer.mode == 'RGB':
                            # Convert RGB to 16-bit
                            img_array = np.array(layer)
                            img_16bit = img_array.astype(np.uint16) * 256
                            img_16bit_rgb = Image.fromarray(img_16bit, mode='RGB')
                            converted_layers.append(img_16bit_rgb)
                            print(f"[TIFF] Converted {name} to 16-bit RGB")
                        
                        elif layer.mode == 'RGBA':
                            # Convert RGBA to 16-bit
                            img_array = np.array(layer)
                            img_16bit = img_array.astype(np.uint16) * 256
                            img_16bit_rgba = Image.fromarray(img_16bit, mode='RGBA')
                            converted_layers.append(img_16bit_rgba)
                            print(f"[TIFF] Converted {name} to 16-bit RGBA")
                        
                        elif layer.mode == 'L':
                            # Convert grayscale to 16-bit
                            img_array = np.array(layer)
                            img_16bit = img_array.astype(np.uint16) * 256
                            img_16bit_gray = Image.fromarray(img_16bit, mode='I;16')
                            converted_layers.append(img_16bit_gray)
                            print(f"[TIFF] Converted {name} to 16-bit grayscale")
                        
                        else:
                            # Keep as-is with warning
                            print(f"[TIFF] Warning: Cannot convert {layer.mode} to 16-bit")
                            converted_layers.append(layer)
                    
                    except Exception as e:
                        print(f"[TIFF] Error converting {name} to 16-bit: {str(e)}")
                        # Keep original
                        converted_layers.append(layer)
                
                # Replace with converted layers
                all_layers = converted_layers
                print(f"[TIFF] Completed 16-bit conversion")
            
            # Add special tags to make Photoshop recognize the layers
            from PIL.TiffTags import TAGS, TYPES
            PHOTOSHOP_LAYERS_TAG = 37724  # Tag number for Photoshop layer data
            
            # Create custom tag dictionary for Photoshop layers
            tags = {}
            
            # Build simple layer information - this is a placeholder
            # Note: Real Photoshop layer data is complex and beyond the scope of a simple implementation
            # This is just to indicate that layers exist
            layer_info = {
                'count': len(all_layers),
                'names': layer_names
            }
            
            # Serialize layer info to bytes
            layer_info_bytes = json.dumps(layer_info).encode('utf-8')
            
            # Add layer info tag
            tags[PHOTOSHOP_LAYERS_TAG] = layer_info_bytes
            
            # Update metadata with layer info
            if metadata is None:
                metadata = {}
            
            if 'analysis' not in metadata:
                metadata['analysis'] = {}
            
            metadata['analysis']['layer_info'] = {
                'layer_count': len(all_layers),
                'layer_names': layer_names
            }
            
            # Add technical info
            if 'technical' not in metadata:
                metadata['technical'] = {}
            
            metadata['technical']['bit_depth'] = bit_depth
            if color_profile_data:
                metadata['technical']['color_profile'] = {
                    'name': color_profile_name,
                    'embedded': True
                }
            
            # Save first image with all the tags and profile
            first_image = all_layers[0]
            first_image.save(
                image_path,
                format="TIFF",
                compression=tiff_compression,
                tiffinfo=tags,
                icc_profile=color_profile_data
            )
            
            # Add remaining images as pages
            if len(all_layers) > 1:
                with Image.open(image_path) as tiff:
                    tiff.save(
                        image_path,
                        format="TIFF",
                        compression=tiff_compression,
                        save_all=True,
                        append_images=all_layers[1:],
                        tiffinfo=tags,
                        icc_profile=color_profile_data
                    )
            
            print(f"[TIFF] Successfully saved TIFF with {len(all_layers)} pages")
            
            # Add metadata with service
            if metadata and kwargs.get("enable_metadata", True):
                try:
                    # Determine targets
                    metadata_targets = []
                    if kwargs.get("save_embedded", True):
                        metadata_targets.append("embedded")
                    if kwargs.get("save_xmp", True):
                        metadata_targets.append("xmp")
                    if kwargs.get("save_txt", True):
                        metadata_targets.append("txt")
                    if kwargs.get("save_db", False):
                        metadata_targets.append("db")
                    
                    if metadata_targets:
                        # Make sure ICC data is stored in metadata
                        if color_profile_data:
                            metadata['__icc_profile_data'] = color_profile_data
                        
                        # Call without icc_profile parameter
                        self.metadata_service.write_metadata(
                            image_path,
                            metadata,
                            targets=metadata_targets
                        )
                        print(f"[TIFF] Added metadata via service")
                except Exception as e:
                    print(f"[TIFF] Error adding metadata: {str(e)}")
            
            return True
                
        except Exception as e:
            print(f"[TIFF] Error saving TIFF: {str(e)}")
            traceback.print_exc()
            return False

    def _metadata_to_xmp_string(self, metadata):
        """
        Convert metadata dictionary to XMP XML string
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            str: XMP packet as XML string
        """
        import xml.etree.ElementTree as ET
        
        # Create base XMP structure
        xmpmeta = ET.Element("{adobe:ns:meta/}xmpmeta")
        
        # Add namespaces
        namespaces = {
            'x': 'adobe:ns:meta/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
            'xmpRights': 'http://ns.adobe.com/xap/1.0/rights/',
            'eiqa': 'http://ericproject.org/schemas/eiqa/1.0/',
            'ai': 'http://ericproject.org/schemas/ai/1.0/'
        }
        
        for prefix, uri in namespaces.items():
            xmpmeta.set(f"xmlns:{prefix}", uri)
        
        # Create RDF element
        rdf = ET.SubElement(xmpmeta, f"{{{namespaces['rdf']}}}RDF")
        
        # Create Description element
        desc = ET.SubElement(rdf, f"{{{namespaces['rdf']}}}Description")
        desc.set(f"{{{namespaces['rdf']}}}about", "")
        
        # Add each metadata section
        for section, section_data in metadata.items():
            if not isinstance(section_data, dict):
                continue
                
            # Map each section to appropriate namespace
            if section == 'basic':
                # Basic fields - map to standard namespaces
                for key, value in section_data.items():
                    if key == 'title':
                        self._add_language_alt(desc, 'dc', 'title', value, namespaces)
                    elif key == 'description':
                        self._add_language_alt(desc, 'dc', 'description', value, namespaces)
                    elif key == 'keywords' and value:
                        self._add_bag(desc, 'dc', 'subject', value, namespaces)
                    elif key == 'creator':
                        self._add_seq(desc, 'dc', 'creator', [value], namespaces)
                    elif key in ('copyright', 'rights'):
                        self._add_language_alt(desc, 'dc', 'rights', value, namespaces)
            
            elif section == 'photoshop':
                # Photoshop fields
                for key, value in section_data.items():
                    elem = ET.SubElement(desc, f"{{{namespaces['photoshop']}}}{key}")
                    elem.text = str(value)
                    
            elif section == 'xmpRights':
                # Rights management fields
                for key, value in section_data.items():
                    elem = ET.SubElement(desc, f"{{{namespaces['xmpRights']}}}{key}")
                    if isinstance(value, bool):
                        elem.text = 'True' if value else 'False'
                    else:
                        elem.text = str(value)
            
            elif section == 'ai_info':
                # AI generation info
                if 'generation' in section_data:
                    gen_data = section_data['generation']
                    gen_elem = ET.SubElement(desc, f"{{{namespaces['ai']}}}generation")
                    gen_desc = ET.SubElement(gen_elem, f"{{{namespaces['rdf']}}}Description")
                    
                    for key, value in gen_data.items():
                        if key == 'loras' and isinstance(value, list):
                            loras_elem = ET.SubElement(desc, f"{{{namespaces['ai']}}}loras")
                            lora_seq = ET.SubElement(loras_elem, f"{{{namespaces['rdf']}}}Seq")
                            
                            for lora in value:
                                li = ET.SubElement(lora_seq, f"{{{namespaces['rdf']}}}li")
                                if isinstance(lora, dict):
                                    lora_desc = ET.SubElement(li, f"{{{namespaces['rdf']}}}Description")
                                    for lora_key, lora_val in lora.items():
                                        lora_elem = ET.SubElement(lora_desc, f"{{{namespaces['ai']}}}{lora_key}")
                                        lora_elem.text = str(lora_val)
                                else:
                                    li.text = str(lora)
                        else:
                            val_elem = ET.SubElement(gen_desc, f"{{{namespaces['ai']}}}{key}")
                            val_elem.text = str(value)
            
            elif section == 'analysis':
                # Analysis data
                for analysis_type, analysis_data in section_data.items():
                    if not isinstance(analysis_data, dict):
                        continue
                        
                    analysis_elem = ET.SubElement(desc, f"{{{namespaces['eiqa']}}}{analysis_type}")
                    analysis_desc = ET.SubElement(analysis_elem, f"{{{namespaces['rdf']}}}Description")
                    
                    for key, value in analysis_data.items():
                        if isinstance(value, dict):
                            # Nested structure
                            sub_elem = ET.SubElement(analysis_desc, f"{{{namespaces['eiqa']}}}{key}")
                            sub_desc = ET.SubElement(sub_elem, f"{{{namespaces['rdf']}}}Description")
                            
                            for subkey, subvalue in value.items():
                                sub_field = ET.SubElement(sub_desc, f"{{{namespaces['eiqa']}}}{subkey}")
                                if isinstance(subvalue, (dict, list)):
                                    sub_field.text = json.dumps(subvalue)
                                else:
                                    sub_field.text = str(subvalue)
                        else:
                            # Simple field
                            field = ET.SubElement(analysis_desc, f"{{{namespaces['eiqa']}}}{key}")
                            if isinstance(value, (dict, list)):
                                field.text = json.dumps(value)
                            else:
                                field.text = str(value)
        
        # Convert to string
        return ET.tostring(xmpmeta, encoding='unicode')

    def _add_language_alt(self, parent, namespace, name, value, namespaces):
        """Add a language alternative structure"""
        elem = ET.SubElement(parent, f"{{{namespaces[namespace]}}}{name}")
        alt = ET.SubElement(elem, f"{{{namespaces['rdf']}}}Alt")
        li = ET.SubElement(alt, f"{{{namespaces['rdf']}}}li")
        li.set('xml:lang', 'x-default')
        li.text = str(value)
        
    def _add_bag(self, parent, namespace, name, values, namespaces):
        """Add a bag structure"""
        elem = ET.SubElement(parent, f"{{{namespaces[namespace]}}}{name}")
        bag = ET.SubElement(elem, f"{{{namespaces['rdf']}}}Bag")
        
        # Convert to list if not already
        if not isinstance(values, (list, tuple)):
            values = [values]
            
        for value in values:
            li = ET.SubElement(bag, f"{{{namespaces['rdf']}}}li")
            li.text = str(value)

    def _add_seq(self, parent, namespace, name, values, namespaces):
        """Add a sequence structure"""
        elem = ET.SubElement(parent, f"{{{namespaces[namespace]}}}{name}")
        seq = ET.SubElement(elem, f"{{{namespaces['rdf']}}}Seq")
        
        # Convert to list if not already
        if not isinstance(values, (list, tuple)):
            values = [values]
            
        for value in values:
            li = ET.SubElement(seq, f"{{{namespaces['rdf']}}}li")
            li.text = str(value)

    def _save_16bit_png(self, image_np, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Dedicated method for 16-bit PNG saving with proper error handling
        
        Args:
            image_np: Numpy image array
            image_path: Path to save the image
            metadata: Metadata to embed
            prompt: ComfyUI workflow prompt data
            extra_pnginfo: Additional PNG info
            **kwargs: Additional parameters
        
        Returns:
            bool: Success status
        """
        # Get parameters
        color_profile_name = kwargs.get('color_profile_name', 'sRGB v4 Appearance')
        png_compression = kwargs.get('png_compression', 6)
        embed_workflow = kwargs.get('embed_workflow', True)
        
        # Get color profile data if name is specified
        color_profile_data = self._get_color_profile(color_profile_name) if color_profile_name != 'None' else None
        
        # Prepare workflow data if needed
        workflow_pnginfo = None
        if embed_workflow and prompt:
            workflow_pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
        
        
        # Ensure image is in 16-bit range (0-65535)
        if image_np.dtype != np.uint16:
            # Convert based on the current range
            if image_np.max() <= 1.0:
                # 0-1 float range
                image_np = (image_np * 65535.0).astype(np.uint16)
            elif image_np.max() <= 255:
                # 8-bit range
                image_np = ((image_np / 255.0) * 65535.0).astype(np.uint16)
            else:
                # Other range - normalize
                max_val = image_np.max()
                image_np = ((image_np / max_val) * 65535.0).astype(np.uint16)
        
        print(f"[MetadataSaveImage] 16-bit conversion - min: {image_np.min()}, max: {image_np.max()}, dtype: {image_np.dtype}")
        
        # Try saving with best available method
        success = False
        
        # Try using pypng for best 16-bit support
        if HAS_PYPNG:
            try:
                success = self._save_16bit_with_pypng(
                    image_np, 
                    image_path, 
                    metadata=metadata,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    color_profile_data=color_profile_data,
                    workflow_pnginfo=workflow_pnginfo,
                    png_compression=png_compression
                )
                if success:
                    print(f"[MetadataSaveImage] Saved 16-bit PNG using pypng: {image_path}")
                    return True
            except Exception as e:
                print(f"[MetadataSaveImage] Error with pypng: {str(e)}, falling back to OpenCV")
        
        # Fall back to OpenCV if pypng unavailable or failed
        if not success:
            try:
                success = self._save_16bit_with_opencv(
                    image_np, 
                    image_path, 
                    metadata=metadata,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    color_profile_data=color_profile_data,
                    workflow_pnginfo=workflow_pnginfo,
                    png_compression=png_compression
                )
                if success:
                    print(f"[MetadataSaveImage] Saved 16-bit PNG using OpenCV: {image_path}")
                    return True
            except Exception as e:
                print(f"[MetadataSaveImage] Error with OpenCV: {str(e)}, falling back to 8-bit")
        
        # If both 16-bit methods fail, try 8-bit as last resort
        if not success:
            try:
                print("[MetadataSaveImage] Falling back to 8-bit PNG...")
                # Convert to 8-bit
                image_8bit = (image_np / 257).astype(np.uint8)  # 65535/257 ≈ 255
                
                # Create PIL image
                img = Image.fromarray(image_8bit)
                
                # Save with metadata
                save_options = {"compress_level": png_compression}
                
                # Add ICC profile if available
                if color_profile_data:
                    save_options["icc_profile"] = color_profile_data
                
                # Save image
                img.save(image_path, format="PNG", pnginfo=workflow_pnginfo, **save_options)
                print(f"[MetadataSaveImage] Saved as 8-bit PNG (16-bit conversion failed): {image_path}")
                return True
            except Exception as e:
                print(f"[MetadataSaveImage] All PNG saving methods failed: {str(e)}")
                return False
        
        return success

    def _save_16bit_with_pypng(self, image_np, image_path, metadata=None, prompt=None, extra_pnginfo=None, color_profile_data=None, workflow_pnginfo=None, png_compression=6, **kwargs):
        """
        Save 16-bit PNG using pypng library
        
        Args:
            image_np: Numpy image array
            image_path: Path to save the image
            metadata: Metadata to embed
            prompt: ComfyUI workflow prompt data
            extra_pnginfo: Additional PNG info
            workflow_pnginfo: Prepared PNG info object with workflow
            color_profile_data: ICC profile data
            png_compression: PNG compression level (0-9)
            **kwargs: Additional parameters
        
        Returns:
            bool: Success status
        """

        # Extract tEXt chunks from workflow_pnginfo
        text_chunks = []
        if workflow_pnginfo:
            try:
                # Create a temporary image
                temp_img = Image.new('RGB', (1, 1))
                # Use BytesIO to create a temporary file
                tmp = io.BytesIO()
                # Save with pnginfo to get the chunks
                temp_img.save(tmp, "png", pnginfo=workflow_pnginfo, compress_level=0)
                tmp.seek(0)
                # Read it back as PNG and get the tEXt chunks
                reader = png.Reader(tmp)
                text_chunks = [x for x in reader.chunks() if x[0] == b"tEXt"]
                print(f"[MetadataSaveImage] Extracted {len(text_chunks)} tEXt chunks for workflow")
            except Exception as e:
                print(f"[MetadataSaveImage] Error extracting tEXt chunks: {str(e)}")

        try:
            # Prepare data for pypng
            height, width, channels = image_np.shape
            
            # Create flattened rows for pypng - much simpler approach
            # Restructure the array to flatrows format that pypng requires
            flat_rows = []
            for i in range(height):
                row = []
                for j in range(width):
                    for c in range(channels):
                        row.append(int(image_np[i, j, c]))
                flat_rows.append(row)
            
            # Create PNG writer
            writer = png.Writer(
                width=width,
                height=height,
                bitdepth=16,
                greyscale=False,
                compression=png_compression
            )
            
            # First write to a memory buffer so we can extract and modify chunks
            buffer = io.BytesIO()
            writer.write(buffer, flat_rows)
            buffer.seek(0)
            
            # Read chunks
            reader = png.Reader(buffer)
            chunks = list(reader.chunks())
            
            # Create a new ordered chunk list
            ordered_chunks = []
            
            # First add IHDR (header) - must be first
            for chunk in chunks:
                if chunk[0] == b'IHDR':
                    ordered_chunks.append(chunk)
                    break
            
            # Add ICC profile right after IHDR if provided
            if color_profile_data:
                try:
                    from zlib import compress
                    profile_name = b'ICC Profile'  # Profile name
                    
                    # PNG spec requires:
                    # 1. Null-terminated profile name
                    # 2. Single compression method byte (0)
                    # 3. Compressed profile data
                    icc_chunk_data = profile_name + b'\0' + b'\0' + compress(color_profile_data)
                    
                    ordered_chunks.append((b'iCCP', icc_chunk_data))
                    print(f"[MetadataSaveImage] Added ICC profile to 16-bit PNG")
                except Exception as e:
                    print(f"[MetadataSaveImage] Error adding ICC profile: {str(e)}")
            
            # Add workflow tEXt chunks right after ICC profile
            for chunk in text_chunks:
                ordered_chunks.append(chunk)
                chunk_type = chunk[0]
                print(f"[MetadataSaveImage] Added tEXt chunk, size: {len(chunk[1])} bytes")
                    
            
            # Now add all remaining chunks except IEND
            for chunk in chunks:
                tag = chunk[0]
                if tag != b'IHDR' and tag != b'iCCP' and tag != b'IEND':
                    ordered_chunks.append(chunk)
            
            # Add IEND last (must be last chunk)
            for chunk in chunks:
                if chunk[0] == b'IEND':
                    ordered_chunks.append(chunk)
                    break
            
            # Write all chunks to file
            with open(image_path, 'wb') as f:
                png.write_chunks(f, ordered_chunks)
            
            return True
            
        except Exception as e:
            print(f"[MetadataSaveImage] Error in _save_16bit_with_pypng: {str(e)}")
            raise  # Re-raise to let parent handle fallback

    def _save_16bit_with_opencv(self, image_np, image_path, metadata=None, prompt=None, extra_pnginfo=None, color_profile_data=None, workflow_pnginfo=None, png_compression=6, **kwargs):
        """
        Save 16-bit PNG using pypng library
        
        Args:
            image_np: Numpy image array
            image_path: Path to save the image
            metadata: Metadata to embed
            prompt: ComfyUI workflow prompt data
            extra_pnginfo: Additional PNG info
            workflow_pnginfo: Prepared PNG info object with workflow
            color_profile_data: ICC profile data
            png_compression: PNG compression level (0-9)
            **kwargs: Additional parameters
        
        Returns:
            bool: Success status
        """
        # Only create a new workflow_pnginfo if none was provided
        if workflow_pnginfo is None and kwargs.get('embed_workflow', True) and prompt:
            workflow_pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
        
        # Extract tEXt chunks from workflow_pnginfo
        text_chunks = []
        if workflow_pnginfo:
            try:
                # Create a temporary image with the workflow data
                temp_img = Image.new('RGB', (1, 1))
                tmp = io.BytesIO()
                temp_img.save(tmp, "png", pnginfo=workflow_pnginfo, compress_level=0)
                tmp.seek(0)
                # Extract the tEXt chunks
                reader = png.Reader(tmp)
                text_chunks = [x for x in reader.chunks() if x[0] == b"tEXt"]
                print(f"[MetadataSaveImage] OpenCV method: Extracted {len(text_chunks)} tEXt chunks")
            except Exception as e:
                print(f"[MetadataSaveImage] Error extracting tEXt chunks: {str(e)}")

        # Convert RGB to BGR for OpenCV
        if image_np.shape[2] == 3:  # RGB input
            opencv_image = cv2.merge((
                image_np[:, :, 2],  # B (from R)
                image_np[:, :, 1],  # G
                image_np[:, :, 0]   # R (from B)
            ))
        else:  # RGBA input
            opencv_image = cv2.merge((
                image_np[:, :, 2],  # B (from R)
                image_np[:, :, 1],  # G
                image_np[:, :, 0],  # R (from B)
                image_np[:, :, 3]   # A
            ))
        
        # Set compression parameters
        params = [
            cv2.IMWRITE_PNG_COMPRESSION, png_compression,
            cv2.IMWRITE_PNG_BILEVEL, 0  # Ensure not bilevel
        ]
        
        # Save with OpenCV first to preserve 16-bit depth
        cv2.imwrite(image_path, opencv_image, params)
        
        # If we need to add metadata, we'll need to use PIL to add it
        # This is tricky because PIL doesn't fully support 16-bit depth
        if color_profile_data or workflow_pnginfo:
            try:
                # Create a temporary file path
                temp_path = image_path + ".temp.png"
                
                # Open the file with PIL, but don't convert bit depth
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    img.load()  # Make sure image is fully loaded
                    
                    # Prepare save options
                    save_options = {}
                    if color_profile_data:
                        save_options["icc_profile"] = color_profile_data
                    
                    # Save to temp file with metadata
                    img.save(temp_path, format="PNG", pnginfo=workflow_pnginfo, **save_options)
                
                # Verify the temp file is valid and replace the original
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    os.replace(temp_path, image_path)
                    print(f"[MetadataSaveImage] Added metadata to OpenCV 16-bit PNG")
                else:
                    # Keep original if temp file is invalid
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    print(f"[MetadataSaveImage] Keeping original 16-bit PNG (metadata update failed)")
            except Exception as e:
                print(f"[MetadataSaveImage] Error adding metadata to 16-bit PNG: {str(e)}")
                # If there's an error, we still have the original file
        
        return True

    def _save_as_psd(self, image, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """Save image as layered PSD with mask and overlay support using psd-tools"""
        try:
            # Convert to PIL image
            pil_image = self._convert_to_pil(image)
            print(f"[PSD] Starting with image: {pil_image.mode}, {pil_image.size}")
            
            # Get dimensions
            width, height = pil_image.size
            
            # Create a new PSD document with RGBA mode to support transparency
            psd = PSDImage.new(mode='RGBA', size=(width, height))
            print(f"[PSD] Created new PSD document in RGBA mode")
            
            # Handle main image - ensure it's in RGBA mode to preserve transparency
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
                print(f"[PSD] Converted main image to RGBA mode")
            
            # Create main layer from RGBA image
            main_layer = PixelLayer.frompil(pil_image, psd)
            main_layer.name = "Main Image"
            psd.append(main_layer)
            print(f"[PSD] Added main image layer with transparency")
            
            # Track layers for metadata
            layers_info = [{"name": "Main Image", "blend_mode": "normal", "opacity": 255}]
            
            # Process overlay layers - ensure all are in RGBA mode
            for i in range(1, 3):  # Up to 2 overlays
                overlay_key = f"overlay_layer_{i}"
                name_key = f"overlay_{i}_name"
                blend_key = f"overlay_{i}_blend_mode"
                opacity_key = f"overlay_{i}_opacity"
                
                if overlay_key in kwargs and kwargs[overlay_key] is not None:
                    try:
                        overlay_img = self._convert_to_pil(kwargs[overlay_key])
                        overlay_name = kwargs.get(name_key, f"Overlay {i}")
                        blend_mode = kwargs.get(blend_key, "normal")
                        opacity = kwargs.get(opacity_key, 255)
                        
                        print(f"[PSD] Processing overlay: {overlay_name}, mode: {overlay_img.mode}")
                        
                        # Ensure overlay is in RGBA mode to preserve transparency
                        if overlay_img.mode != 'RGBA':
                            overlay_img = overlay_img.convert('RGBA')
                            print(f"[PSD] Converted overlay {i} to RGBA mode")
                        
                        # Create the overlay layer
                        overlay_layer = PixelLayer.frompil(overlay_img, psd)
                        overlay_layer.name = overlay_name
                        
                        # Apply settings
                        try:
                            # Set blend mode
                            blend_mode_value = getattr(BlendMode, blend_mode.upper())
                            overlay_layer.blend_mode = blend_mode_value
                            
                            # Set opacity (0-255)
                            if isinstance(opacity, (int, float)):
                                opacity_value = int(min(max(opacity, 0), 255))
                                overlay_layer.opacity = opacity_value
                                print(f"[PSD] Set opacity {opacity_value} for {overlay_name}")
                            
                            print(f"[PSD] Set blend mode {blend_mode} for {overlay_name}")
                        except AttributeError as e:
                            print(f"[PSD] Warning: {str(e)}")
                        
                        # Add layer
                        psd.append(overlay_layer)
                        
                        # Track info
                        layers_info.append({
                            "name": overlay_name,
                            "blend_mode": blend_mode,
                            "opacity": overlay_layer.opacity,
                            "has_transparency": True  # Always True since we converted to RGBA
                        })
                        
                        print(f"[PSD] Added overlay layer: {overlay_name}")
                    except Exception as e:
                        print(f"[PSD] Error adding overlay {i}: {str(e)}")
            
            # Process mask layers
            for i in range(1, 3):  # Up to 2 masks
                mask_key = f"mask_layer_{i}"
                name_key = f"mask_{i}_name"
                blend_key = f"mask_{i}_blend_mode"
                opacity_key = f"mask_{i}_opacity"
                
                if mask_key in kwargs and kwargs[mask_key] is not None:
                    try:
                        mask_img = self._convert_to_pil(kwargs[mask_key])
                        mask_name = kwargs.get(name_key, f"Masked Layer {i}")
                        blend_mode = kwargs.get(blend_key, "normal")
                        opacity = kwargs.get(opacity_key, 255)
                        
                        # Ensure the mask is grayscale
                        if mask_img.mode != 'L':
                            mask_img = mask_img.convert('L')
                        
                        # Get a duplicate of the main image in RGBA mode
                        if pil_image.mode != 'RGBA':
                            base_img = pil_image.convert('RGBA')
                        else:
                            base_img = pil_image.copy()
                        
                        # Apply the mask as alpha channel
                        # First, split the channels
                        r, g, b, a = base_img.split()
                        
                        # Combine original alpha with mask (using mask to further reduce alpha)
                        new_alpha = ImageChops.multiply(a, mask_img)
                        
                        # Merge back with the new alpha
                        masked_img = Image.merge('RGBA', (r, g, b, new_alpha))
                        
                        # Create a new layer with the masked image
                        masked_layer = PixelLayer.frompil(masked_img, psd)
                        masked_layer.name = mask_name
                        
                        # Set blend mode
                        try:
                            blend_mode_value = getattr(BlendMode, blend_mode.upper())
                            masked_layer.blend_mode = blend_mode_value
                        except AttributeError as e:
                            print(f"[PSD] Warning: {str(e)}")
                        
                        # Set opacity
                        if isinstance(opacity, (int, float)):
                            opacity_value = int(min(max(opacity, 0), 255))
                            masked_layer.opacity = opacity_value
                        
                        # Set visibility to false
                        masked_layer.visible = False
                        
                        # Add layer
                        psd.append(masked_layer)
                        
                        # Track info
                        layers_info.append({
                            "name": mask_name,
                            "blend_mode": blend_mode,
                            "opacity": masked_layer.opacity,
                            "visible": False
                        })
                        
                        print(f"[PSD] Added masked version of main image: {mask_name}")
                    except Exception as e:
                        print(f"[PSD] Error adding masked layer {i}: {str(e)}")
            
            # Add bit depth info to metadata
            bit_depth = kwargs.get("bit_depth", "8-bit")
            is_16bit = bit_depth == "16-bit"
            
            if metadata is None:
                metadata = {}
            
            if 'technical' not in metadata:
                metadata['technical'] = {}
            
            metadata['technical']['bit_depth'] = bit_depth
            
            # Add layer info to metadata
            if 'analysis' not in metadata:
                metadata['analysis'] = {}
            
            metadata['analysis']['layer_info'] = {
                'layer_count': len(layers_info),
                'layers': layers_info
            }
            
            # Get color profile info
            color_profile_name = kwargs.get("color_profile", "sRGB v4 Appearance")
            color_profile_data = self._get_color_profile(color_profile_name)
            
            if color_profile_data:
                metadata['technical']['color_profile'] = {
                    'name': color_profile_name,
                    'embedded': True
                }
                
                # Store in metadata directly
                metadata['__icc_profile_data'] = color_profile_data
                
                # Add ICC profile to PSD directly as image resource
                from psd_tools.constants import Resource
                from psd_tools.psd.image_resources import ImageResource, ImageResources
                
                # Ensure image_resources exists
                if not hasattr(psd._record, 'image_resources'):
                    psd._record.image_resources = ImageResources([])
                
                # Add ICC profile - dictionary style assignment
                psd._record.image_resources[1039] = ImageResource(
                    signature=b'8BIM',
                    key=1039,  # Resource.ICC_PROFILE.value
                    name='',
                    data=color_profile_data
                )
                
                print(f"[PSD] Added ICC profile: {color_profile_name}")
            
            # Add metadata directly to PSD before saving
            if metadata and kwargs.get("enable_metadata", True):
                try:
                    # Generate XMP metadata string
                    xmp_str = self._metadata_to_xmp_string(metadata)
                    xmp_bytes = xmp_str.encode('utf-8')
                    
                    # Add XMP metadata - dictionary style assignment
                    psd._record.image_resources[1060] = ImageResource(
                        signature=b'8BIM',
                        key=1060,  # Resource.XMP_METADATA.value
                        name='',
                        data=xmp_bytes
                    )
                        
                    # Generate IPTC data 
                    iptc_data = self._generate_iptc_data(metadata)
                    if iptc_data:
                        # Add IPTC data - dictionary style assignment
                        psd._record.image_resources[1028] = ImageResource(
                            signature=b'8BIM',
                            key=1028,  # Resource.IPTC_NAA.value
                            name='',
                            data=iptc_data
                        )
                    
                    print(f"[PSD] Added XMP and IPTC metadata to PSD")
                except Exception as meta_err:
                    print(f"[PSD] Error adding metadata to PSD: {str(meta_err)}")
            
            # Save the PSD with all resources and metadata
            print(f"[PSD] Saving to: {image_path}")
            psd.save(image_path)
            print(f"[PSD] Saved PSD with {len(psd)} layers")
            
            # Still create sidecar files as needed
            if metadata and kwargs.get("enable_metadata", True):
                try:
                    # Determine targets for sidecar metadata
                    metadata_targets = []
                    if kwargs.get("save_xmp", True):
                        metadata_targets.append("xmp")  # XMP sidecar
                    if kwargs.get("save_txt", True):
                        metadata_targets.append("txt")  # Text file
                    if kwargs.get("save_db", False):
                        metadata_targets.append("db")   # Database
                    
                    if metadata_targets:
                        # Call metadata service for sidecar files
                        self.metadata_service.write_metadata(
                            image_path,
                            metadata,
                            targets=metadata_targets
                        )
                        print(f"[PSD] Added metadata sidecar files: {', '.join(metadata_targets)}")
                except Exception as e:
                    print(f"[PSD] Error creating metadata sidecar files: {str(e)}")
            
            # Note about bit depth limitation
            if is_16bit:
                print("[PSD] Note: 16-bit PSD is not fully supported by psd-tools.")
                print("[PSD] Consider using TIFF format for 16-bit images.")
            
            return True
            
        except Exception as e:
            print(f"[PSD] Error in _save_as_psd: {str(e)}")
            traceback.print_exc()
            
            return False

    def _generate_iptc_data(self, metadata):
        """
        Generate simplified IPTC data from metadata dictionary
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            bytes: IPTC data or None if not able to generate
        """
        try:
            # Create a minimal IPTC data structure
            # This is a very simplified implementation
            iptc_data = bytearray()
            
            # Extract basic fields from metadata
            title = ""
            description = ""
            creator = ""
            keywords = []
            copyright_notice = ""
            
            # Extract from our standard metadata structure
            if 'basic' in metadata:
                basic = metadata['basic']
                title = basic.get('title', '')
                description = basic.get('description', '')
                creator = basic.get('creator', '')
                copyright_notice = basic.get('copyright', '') or basic.get('rights', '')
                
                if 'keywords' in basic:
                    if isinstance(basic['keywords'], list):
                        keywords = basic['keywords']
                    elif isinstance(basic['keywords'], str):
                        keywords = [k.strip() for k in basic['keywords'].split(',') if k.strip()]
            
            # IPTC markers and tags
            IPTC_RECORD_VERSION = b'\x1c\x01'
            IPTC_OBJECT_NAME = b'\x1c\x05'  # Title
            IPTC_CAPTION = b'\x1c\x78'  # Description
            IPTC_BYLINE = b'\x1c\x50'  # Creator
            IPTC_KEYWORDS = b'\x1c\x19'
            IPTC_COPYRIGHT = b'\x1c\x74'
            
            # Add record version
            iptc_data.extend(IPTC_RECORD_VERSION)
            iptc_data.extend(b'\x00\x02')  # Length 2
            iptc_data.extend(b'\x00\x02')  # Version 2
            
            # Add title
            if title:
                iptc_data.extend(IPTC_OBJECT_NAME)
                title_bytes = title.encode('utf-8')
                title_len = len(title_bytes)
                iptc_data.extend(bytes([0, title_len]))  # Length
                iptc_data.extend(title_bytes)
            
            # Add description
            if description:
                iptc_data.extend(IPTC_CAPTION)
                desc_bytes = description.encode('utf-8')
                desc_len = len(desc_bytes)
                iptc_data.extend(bytes([0, desc_len]))  # Length
                iptc_data.extend(desc_bytes)
            
            # Add creator
            if creator:
                iptc_data.extend(IPTC_BYLINE)
                creator_bytes = creator.encode('utf-8')
                creator_len = len(creator_bytes)
                iptc_data.extend(bytes([0, creator_len]))  # Length
                iptc_data.extend(creator_bytes)
            
            # Add keywords
            for keyword in keywords:
                if keyword:
                    iptc_data.extend(IPTC_KEYWORDS)
                    keyword_bytes = keyword.encode('utf-8')
                    keyword_len = len(keyword_bytes)
                    iptc_data.extend(bytes([0, keyword_len]))  # Length
                    iptc_data.extend(keyword_bytes)
            
            # Add copyright
            if copyright_notice:
                iptc_data.extend(IPTC_COPYRIGHT)
                copyright_bytes = copyright_notice.encode('utf-8')
                copyright_len = len(copyright_bytes)
                iptc_data.extend(bytes([0, copyright_len]))  # Length
                iptc_data.extend(copyright_bytes)
            
            return bytes(iptc_data) if iptc_data else None
        except Exception as e:
            print(f"[PSD] Error generating IPTC data: {str(e)}")
            return None

    def _detect_layers_and_adjust_format(self, current_format, **kwargs):
        """
        Detect if layers are present and suggest or adjust format accordingly
        
        Args:
            current_format: Current selected file format
            **kwargs: Node parameters including layer inputs
            
        Returns:
            tuple: (adjusted_format, format_changed, message)
        """
        # Check if any layer inputs are provided
        has_mask_layers = False
        has_overlay_layers = False
        
        # Check for mask layers
        for i in range(1, 3):
            mask_key = f"mask_layer_{i}"
            if mask_key in kwargs and kwargs[mask_key] is not None:
                has_mask_layers = True
                break
        
        # Check for overlay layers
        for i in range(1, 3):
            overlay_key = f"overlay_layer_{i}"
            if overlay_key in kwargs and kwargs[overlay_key] is not None:
                has_overlay_layers = True
                break
        
        # If no layers, keep the current format
        if not has_mask_layers and not has_overlay_layers:
            return current_format, False, ""
        
        # Define formats that support layers
        layer_formats = ["tiff", "psd"]
        
        # If current format already supports layers, keep it
        if current_format.lower() in layer_formats:
            return current_format, False, ""
        
        # If layers detected but format doesn't support layers, suggest or adjust
        message = (f"[MetadataSaveImage] Layers detected but {current_format.upper()} format doesn't support layers. "
                f"Consider using TIFF or PSD format for proper layer support.")
        
        # Option 1: Auto-switch (commented out by default)
        # adjusted_format = "tiff"  # Default to TIFF
        # message = f"[MetadataSaveImage] Layers detected - automatically switched from {current_format.upper()} to {adjusted_format.upper()}"
        # return adjusted_format, True, message
        
        # Option 2: Just warn but keep original format (default behavior)
        print(message)
        return current_format, False, message

    def _save_as_svg(self, image, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Convert and save image as SVG with metadata
        
        Args:
            image: Image tensor or array
            image_path: Path to save the SVG
            metadata: Metadata to embed in the SVG
            **kwargs: SVG-specific parameters
            
        Returns:
            bool: Success status
        """
        if not HAS_VTRACER:
            print("[MetadataSaveImage] vtracer not installed - SVG export not available")
            return False
            
        try:
            # Get SVG parameters
            colormode = kwargs.get("svg_colormode", "color")
            hierarchical = kwargs.get("svg_hierarchical", "stacked")
            mode = kwargs.get("svg_mode", "spline")
            
            # Convert image to PIL first
            pil_image = self._convert_to_pil(image)
            
            # Make sure we have an alpha channel
            if pil_image.mode != 'RGBA':
                alpha = Image.new('L', pil_image.size, 255)
                pil_image.putalpha(alpha)
            
            # Extract pixels and size
            pixels = list(pil_image.getdata())
            size = pil_image.size
            
            # Prepare SVG conversion parameters
            params = {
                'colormode': colormode,
                'hierarchical': hierarchical,
                'mode': mode,
                'filter_speckle': kwargs.get("svg_filter_speckle", 4),
                'color_precision': kwargs.get("svg_color_precision", 6),
                'layer_difference': kwargs.get("svg_layer_difference", 16),
                'corner_threshold': kwargs.get("svg_corner_threshold", 60),
                'length_threshold': kwargs.get("svg_length_threshold", 4.0),
                'max_iterations': kwargs.get("svg_max_iterations", 10),
                'splice_threshold': kwargs.get("svg_splice_threshold", 45),
                'path_precision': kwargs.get("svg_path_precision", 3)
            }
            
            # Convert to SVG
            svg_string = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                **params
            )
            
            # Add metadata to SVG if provided
            if metadata:
                svg_string = self._add_metadata_to_svg(svg_string, metadata)
            
            # Write SVG file
            with open(image_path, 'w', encoding='utf-8') as f:
                f.write(svg_string)
                
            print(f"[MetadataSaveImage] Saved image as SVG: {image_path}")
            return True
        
        except Exception as e:
            print(f"[MetadataSaveImage] Error saving as SVG: {str(e)}")
            return False

    def _add_metadata_to_svg(self, svg_string, metadata):
        """
        Add metadata to an SVG file
        
        Args:
            svg_string: SVG content as string
            metadata: Metadata to add
            
        Returns:
            str: Updated SVG content
        """
        try:
            # Parse SVG content
            import xml.etree.ElementTree as ET
            
            # Parse with namespace support
            ET.register_namespace('', "http://www.w3.org/2000/svg")
            ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
            
            svg_tree = ET.ElementTree(ET.fromstring(svg_string))
            svg_root = svg_tree.getroot()
            
            # Create metadata element if not present
            metadata_elem = svg_root.find('.//{http://www.w3.org/2000/svg}metadata')
            if metadata_elem is None:
                metadata_elem = ET.SubElement(svg_root, '{http://www.w3.org/2000/svg}metadata')
            
            # Create XMP metadata
            # Register all needed namespaces
            namespaces = {
                'x': 'adobe:ns:meta/',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'xmp': 'http://ns.adobe.com/xap/1.0/',
                'dc': 'http://purl.org/dc/elements/1.1/',
                'photoshop': 'http://ns.adobe.com/photoshop/1.0/',
                'xmpRights': 'http://ns.adobe.com/xap/1.0/rights/',
                'eiqa': 'http://ericproject.org/schemas/eiqa/1.0/',
                'ai': 'http://ericproject.org/schemas/ai/1.0/',
                'Iptc4xmpCore': 'http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/',
                'plus': 'http://ns.useplus.org/ldf/xmp/1.0/'
            }
            
            for prefix, uri in namespaces.items():
                ET.register_namespace(prefix, uri)
            
            # Create the RDF structure
            rdf_root = ET.SubElement(metadata_elem, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
            description = ET.SubElement(rdf_root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
            description.set('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', "")
            
            # Add metadata fields by section
            for section, section_data in metadata.items():
                if not isinstance(section_data, dict):
                    continue
                    
                # Process based on section type
                if section == 'basic':
                    self._add_basic_metadata_to_svg(description, section_data, namespaces)
                elif section == 'ai_info':
                    self._add_ai_metadata_to_svg(description, section_data, namespaces)
                elif section == 'analysis':
                    self._add_analysis_metadata_to_svg(description, section_data, namespaces)
                elif section == 'dc':
                    self._add_dc_metadata_to_svg(description, section_data, namespaces)
                elif section == 'photoshop':
                    self._add_photoshop_metadata_to_svg(description, section_data, namespaces)
                elif section == 'xmpRights':
                    self._add_xmprights_metadata_to_svg(description, section_data, namespaces)
                elif section == 'Iptc4xmpCore':
                    self._add_iptc_metadata_to_svg(description, section_data, namespaces)
            
            # Convert back to string with proper namespaces
            svg_string = ET.tostring(svg_root, encoding='unicode')
            
            return svg_string
        except Exception as e:
            print(f"[MetadataSaveImage] Error adding metadata to SVG: {str(e)}")
            return svg_string  # Return original if there was an error

    def _add_basic_metadata_to_svg(self, parent, basic_data, namespaces):
        """Add basic metadata fields to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        # Handle title
        if 'title' in basic_data:
            title_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}title')
            self._add_language_alt_to_svg(title_elem, basic_data['title'], namespaces)
        
        # Handle description
        if 'description' in basic_data:
            desc_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}description')
            self._add_language_alt_to_svg(desc_elem, basic_data['description'], namespaces)
        
        # Handle creator
        if 'creator' in basic_data:
            creator_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}creator')
            self._add_seq_to_svg(creator_elem, [basic_data['creator']], namespaces)
        
        # Handle keywords
        if 'keywords' in basic_data:
            subject_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}subject')
            self._add_bag_to_svg(subject_elem, basic_data['keywords'], namespaces)
        
        # Handle rights/copyright
        if 'rights' in basic_data:
            rights_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}rights')
            self._add_language_alt_to_svg(rights_elem, basic_data['rights'], namespaces)
        elif 'copyright' in basic_data:
            rights_elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}rights')
            self._add_language_alt_to_svg(rights_elem, basic_data['copyright'], namespaces)
        
        # Handle rating
        if 'rating' in basic_data:
            rating_elem = ET.SubElement(parent, f'{{{namespaces["xmp"]}}}Rating')
            rating_elem.text = str(basic_data['rating'])

    def _add_ai_metadata_to_svg(self, parent, ai_data, namespaces):
        """Add AI generation metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        # Handle generation data
        if 'generation' in ai_data:
            gen_data = ai_data['generation']
            gen_elem = ET.SubElement(parent, f'{{{namespaces["ai"]}}}generation')
            gen_desc = ET.SubElement(gen_elem, f'{{{namespaces["rdf"]}}}Description')
            
            # Add generation fields
            for key, value in gen_data.items():
                if key == 'loras' and isinstance(value, list):
                    # Handle LoRAs as a sequence
                    loras_elem = ET.SubElement(gen_desc, f'{{{namespaces["ai"]}}}loras')
                    seq = ET.SubElement(loras_elem, f'{{{namespaces["rdf"]}}}Seq')
                    
                    for lora in value:
                        li = ET.SubElement(seq, f'{{{namespaces["rdf"]}}}li')
                        lora_desc = ET.SubElement(li, f'{{{namespaces["rdf"]}}}Description')
                        
                        name_elem = ET.SubElement(lora_desc, f'{{{namespaces["ai"]}}}name')
                        name_elem.text = lora.get('name', '')
                        
                        strength_elem = ET.SubElement(lora_desc, f'{{{namespaces["ai"]}}}strength')
                        strength_elem.text = str(lora.get('strength', 1.0))
                else:
                    # Simple field
                    field_elem = ET.SubElement(gen_desc, f'{{{namespaces["ai"]}}}{key}')
                    if isinstance(value, (dict, list)):
                        field_elem.text = json.dumps(value)
                    else:
                        field_elem.text = str(value)

    def _add_analysis_metadata_to_svg(self, parent, analysis_data, namespaces):
        """Add analysis metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        # Process each analysis type
        for analysis_type, data in analysis_data.items():
            if not isinstance(data, dict):
                continue
                
            # Create container for this analysis type
            analysis_elem = ET.SubElement(parent, f'{{{namespaces["eiqa"]}}}{analysis_type}')
            analysis_desc = ET.SubElement(analysis_elem, f'{{{namespaces["rdf"]}}}Description')
            
            # Add data fields
            for key, value in data.items():
                if isinstance(value, dict):
                    # Nested structure
                    sub_elem = ET.SubElement(analysis_desc, f'{{{namespaces["eiqa"]}}}{key}')
                    sub_desc = ET.SubElement(sub_elem, f'{{{namespaces["rdf"]}}}Description')
                    
                    for subkey, subvalue in value.items():
                        sub_field = ET.SubElement(sub_desc, f'{{{namespaces["eiqa"]}}}{subkey}')
                        if isinstance(subvalue, (dict, list)):
                            sub_field.text = json.dumps(subvalue)
                        else:
                            sub_field.text = str(subvalue)
                else:
                    # Simple field
                    field = ET.SubElement(analysis_desc, f'{{{namespaces["eiqa"]}}}{key}')
                    if isinstance(value, (dict, list)):
                        field.text = json.dumps(value)
                    else:
                        field.text = str(value)

    def _add_dc_metadata_to_svg(self, parent, dc_data, namespaces):
        """Add Dublin Core metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        for key, value in dc_data.items():
            # Handle different Dublin Core structures
            if key == 'title' or key == 'description' or key == 'rights':
                if isinstance(value, dict) and 'x-default' in value:
                    # Language alternative structure
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    self._add_language_alt_to_svg(elem, value['x-default'], namespaces)
                else:
                    # Simple value
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    elem.text = str(value)
            elif key == 'creator' or key == 'contributor':
                # Sequence structure
                if isinstance(value, list):
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    self._add_seq_to_svg(elem, value, namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    self._add_seq_to_svg(elem, [value], namespaces)
            elif key == 'subject':
                # Bag structure for keywords
                if isinstance(value, list):
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    self._add_bag_to_svg(elem, value, namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                    self._add_bag_to_svg(elem, [value], namespaces)
            else:
                # Other fields
                elem = ET.SubElement(parent, f'{{{namespaces["dc"]}}}{key}')
                if isinstance(value, (dict, list)):
                    elem.text = json.dumps(value)
                else:
                    elem.text = str(value)
 
    def _add_photoshop_metadata_to_svg(self, parent, photoshop_data, namespaces):
        """Add Photoshop metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        for key, value in photoshop_data.items():
            # All Photoshop fields are simple
            elem = ET.SubElement(parent, f'{{{namespaces["photoshop"]}}}{key}')
            if isinstance(value, (dict, list)):
                elem.text = json.dumps(value)
            elif isinstance(value, bool):
                elem.text = 'True' if value else 'False'
            else:
                elem.text = str(value)

    def _add_xmprights_metadata_to_svg(self, parent, rights_data, namespaces):
        """Add XMP Rights metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        for key, value in rights_data.items():
            if key == 'UsageTerms':
                # Language alternative
                if isinstance(value, dict) and 'x-default' in value:
                    elem = ET.SubElement(parent, f'{{{namespaces["xmpRights"]}}}{key}')
                    self._add_language_alt_to_svg(elem, value['x-default'], namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["xmpRights"]}}}{key}')
                    self._add_language_alt_to_svg(elem, str(value), namespaces)
            elif key == 'Owner':
                # Bag structure
                if isinstance(value, list):
                    elem = ET.SubElement(parent, f'{{{namespaces["xmpRights"]}}}{key}')
                    self._add_bag_to_svg(elem, value, namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["xmpRights"]}}}{key}')
                    self._add_bag_to_svg(elem, [value], namespaces)
            else:
                # Other fields
                elem = ET.SubElement(parent, f'{{{namespaces["xmpRights"]}}}{key}')
                if isinstance(value, (dict, list)):
                    elem.text = json.dumps(value)
                elif isinstance(value, bool):
                    elem.text = 'True' if value else 'False'
                else:
                    elem.text = str(value)

    def _add_iptc_metadata_to_svg(self, parent, iptc_data, namespaces):
        """Add IPTC Core metadata to SVG Description element"""
        import xml.etree.ElementTree as ET
        
        for key, value in iptc_data.items():
            if key in ('Title', 'Description', 'CopyrightNotice'):
                # Language alternative fields
                if isinstance(value, dict) and 'x-default' in value:
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_language_alt_to_svg(elem, value['x-default'], namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_language_alt_to_svg(elem, str(value), namespaces)
            elif key in ('Creator', 'SubjectCode'):
                # Sequence fields
                if isinstance(value, list):
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_seq_to_svg(elem, value, namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_seq_to_svg(elem, [value], namespaces)
            elif key == 'Keywords':
                # Bag structure
                if isinstance(value, list):
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_bag_to_svg(elem, value, namespaces)
                else:
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    self._add_bag_to_svg(elem, [value], namespaces)
            elif key == 'CreatorContactInfo':
                # Structured contact info
                if isinstance(value, dict):
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    contact_desc = ET.SubElement(elem, f'{{{namespaces["rdf"]}}}Description')
                    
                    for contact_key, contact_value in value.items():
                        contact_field = ET.SubElement(contact_desc, f'{{{namespaces["Iptc4xmpCore"]}}}{contact_key}')
                        contact_field.text = str(contact_value)
                else:
                    # Fallback for simple value
                    elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                    elem.text = str(value)
            else:
                # Other IPTC fields
                elem = ET.SubElement(parent, f'{{{namespaces["Iptc4xmpCore"]}}}{key}')
                if isinstance(value, (dict, list)):
                    elem.text = json.dumps(value)
                else:
                    elem.text = str(value)

    # Helper methods for XMP structures
    def _add_language_alt_to_svg(self, parent, value, namespaces):
        """Add language alternative structure to SVG element"""
        import xml.etree.ElementTree as ET
        
        alt = ET.SubElement(parent, f'{{{namespaces["rdf"]}}}Alt')
        li = ET.SubElement(alt, f'{{{namespaces["rdf"]}}}li')
        li.set('xml:lang', 'x-default')
        li.text = str(value)

    def _add_seq_to_svg(self, parent, values, namespaces):
        """Add sequence structure to SVG element"""
        import xml.etree.ElementTree as ET
        
        seq = ET.SubElement(parent, f'{{{namespaces["rdf"]}}}Seq')
        for item in values:
            li = ET.SubElement(seq, f'{{{namespaces["rdf"]}}}li')
            li.text = str(item)

    def _add_bag_to_svg(self, parent, values, namespaces):
        """Add bag structure to SVG element"""
        import xml.etree.ElementTree as ET
        
        bag = ET.SubElement(parent, f'{{{namespaces["rdf"]}}}Bag')
        for item in values:
            li = ET.SubElement(bag, f'{{{namespaces["rdf"]}}}li')
            li.text = str(item)


    def save_with_metadata(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        # Set debug mode from input parameter
        self.debug = kwargs.get("debug_logging", False)
        
        # Ensure filename_prefix parameter is used
        print(f"[DEBUG] save_with_metadata got filename_prefix: {filename_prefix}")
        
        # Update debugging for service objects
        self.metadata_service.debug = self.debug
        self.workflow_parser.debug = self.debug
        self.workflow_extractor.debug = self.debug
        self.workflow_processor.debug = self.debug
        
        if self.debug:
            print(f"[MetadataSaveImage] Starting image save process")
        
        # Store the original images for returning later
        original_images = images
        
        # Results storage
        results = []
        errors = []
        ui_results = []
        
        # Prepare workflow png info
        workflow_pnginfo = None
        embed_workflow = kwargs.get('embed_workflow', True)
        if prompt:
            workflow_pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo, embed_workflow)

        try:
            # Create a copy for processing if it's a tensor
            if torch.is_tensor(images):
                images_for_saving = images.cpu().numpy()
            else:
                images_for_saving = images
            
            # Clean any non-serializable objects from prompts/metadata
            prompt = self._ensure_serializable_metadata(prompt) if prompt else None
            extra_pnginfo = self._ensure_serializable_metadata(extra_pnginfo) if extra_pnginfo else None
        
            # Add filename_prefix to kwargs to ensure it's used in resolve_output_path
            kwargs["filename_prefix"] = filename_prefix
        
            # Resolve output path
            try:
                path_info = self.resolve_output_path(**kwargs)
                output_dir = path_info["output_dir"]
                filename_prefix = path_info["filename_prefix"]
                subfolder = path_info["subfolder"]
                base_output_dir = path_info["base_output_dir"]
            except Exception as path_error:
                # Fall back to default output directory
                output_dir = self.output_dir
                base_output_dir = self.output_dir
                subfolder = ""
                errors.append(f"Path resolution error: {str(path_error)}")
                print(f"[MetadataSaveImage] Error resolving path: {str(path_error)}, using default output directory")
            
            # Build metadata targets list from individual toggles
            metadata_targets = []
            if kwargs.get("save_embedded", True):
                metadata_targets.append("embedded")
            if kwargs.get("save_xmp", True):
                metadata_targets.append("xmp")
            if kwargs.get("save_txt", True):
                metadata_targets.append("txt")
            if kwargs.get("save_db", False):
                metadata_targets.append("db")
            
            
            # Prepare metadata if enabled
            metadata = {}
            if kwargs.get("enable_metadata", True):
                try:
                    # Build metadata from inputs first
                    input_metadata = self.build_metadata_from_inputs(**kwargs)
                    
                    # Extract workflow metadata ONLY ONCE
                    workflow_metadata = {}
                    if prompt or extra_pnginfo:
                        if self.debug:
                            print("[MetadataSaveImage] Extracting metadata from workflow")
                        
                        # Use the workflow processor to extract metadata
                        # IMPORTANT: Always set workflow_in_xmp to false to prevent bloated XMP
                        workflow_metadata = self.extract_metadata_from_workflow(
                            prompt, 
                            extra_pnginfo, 
                            embed_workflow_in_xmp=False,  # Never embed the full workflow in XMP
                            **kwargs
                        )
                        
                        if self.debug and workflow_metadata:
                            print(f"[MetadataSaveImage] Successfully extracted workflow metadata")
                    
                    # Merge with priority for user inputs
                    metadata = self.metadata_service._merge_metadata(workflow_metadata, input_metadata)
                    
                    # Ensure we don't have duplicate workflow data in multiple places
                    for section in ['ai_info', 'workflow', 'prompt', 'workflow_data']:
                        if section in metadata:
                            # For ai_info section, remove workflow data but keep other ai info
                            if section == 'ai_info':
                                if 'generation' in metadata['ai_info']:
                                    # Remove workflow data from generation
                                    for key in ['prompt', 'workflow', 'workflow_data']:
                                        metadata['ai_info']['generation'].pop(key, None)
                            else:
                                # Remove the entire section for other workflow sections
                                metadata.pop(section, None)
                    
                except Exception as metadata_error:
                    errors.append(f"Metadata preparation error: {str(metadata_error)}")
                    print(f"[MetadataSaveImage] Error preparing metadata: {str(metadata_error)}")
                    metadata = {}
            
            # Format-specific parameters
            format_params = self._get_format_parameters(**kwargs)
            
            # Get file format and extension
            file_format = kwargs.get("file_format", "png").lower()

            # Check for layers and potentially adjust format
            file_format, format_changed, layer_message = self._detect_layers_and_adjust_format(current_format=file_format, **kwargs)
            if format_changed:
                print(layer_message)
                # Update the file extension to match the new format
                file_ext = f".{file_format}"
            else:
                file_ext = f".{file_format}"
            
            # Check for alpha channel and potentially adjust format
            file_format, format_changed, layer_message = self._detect_layers_and_adjust_format(current_format=file_format, **kwargs)
            if format_changed:
                print(alpha_message)
                # Update the file extension to match the new format
                file_ext = f".{file_format}"
            else:
                file_ext = f".{file_format}"
            # Process each image in batch
            for img_index, image in enumerate(images_for_saving):
                try:
                    # Convert from BHWC to HWC if needed
                    if len(image.shape) == 4:
                        image = image[0]
                    
                    # Generate filename
                    simple_format = kwargs.get("filename_format") == "Simple (file1.png)"
                    filename, counter = self.generate_sequential_filename(
                        output_dir, 
                        filename_prefix, 
                        file_ext,
                        simple_format=simple_format,
                        counter=img_index
                    )
                    
                    # Create full output path
                    image_path = os.path.join(output_dir, filename)
                    
                    # Save based on format
                    success = False
                    save_error = None
                    
                    try:
                        # Handle SVG specially
                        if file_format == "svg":
                            if HAS_VTRACER:
                                success = self._save_as_svg(image, image_path, metadata=metadata, prompt=prompt, extra_pnginfo=extra_pnginfo, **kwargs)
                            else:
                                save_error = "Cannot save as SVG - vtracer not installed"
                        
                        # Handle 16-bit PNG
                        elif file_format == "png" and kwargs.get("bit_depth", "8-bit") == "16-bit":
                            success = self._save_16bit_png(
                                image, 
                                image_path, 
                                metadata=metadata,
                                prompt=prompt, 
                                extra_pnginfo=extra_pnginfo, 
                                **format_params
                            )

                        # Handle TIFF
                        elif file_format in ["tiff", "tif"]:
                            success = self._save_as_tiff(
                                image, 
                                image_path, 
                                metadata=metadata, 
                                prompt=prompt, 
                                extra_pnginfo=extra_pnginfo, 
                                **kwargs  # Pass all kwargs for layer info
                            )
                        # Handle PSD
                        elif file_format == "psd":
                            success = self._save_as_psd(
                                image, 
                                image_path, 
                                metadata=metadata, 
                                prompt=prompt, 
                                extra_pnginfo=extra_pnginfo, 
                                **kwargs  # Pass all kwargs for layer info
                            )

                        # Handle APNG
                        elif file_format == "apng":
                            # For APNG, check if we have a batch of images
                            is_batch = False
                            if torch.is_tensor(images_for_saving) and len(images_for_saving.shape) == 4 and images_for_saving.shape[0] > 1:
                                # This is a batch of images - use as animation frames
                                is_batch = True
                                success = self._save_as_apng(
                                    images_for_saving, 
                                    image_path, 
                                    metadata=metadata, 
                                    prompt=prompt, 
                                    extra_pnginfo=extra_pnginfo, 
                                    **format_params
                                )
                            else:
                                # Single image - not ideal for APNG but we can still save it
                                print("[MetadataSaveImage] Warning: Saving single frame as APNG")
                                success = self._save_as_apng(
                                    [images_for_saving], 
                                    image_path, 
                                    metadata=metadata, 
                                    prompt=prompt, 
                                    extra_pnginfo=extra_pnginfo, 
                                    **format_params
                                )

                        # For formats that need advanced alpha handling
                        elif file_format in ["jpg", "jpeg"]:
                            success, alpha_path = self._save_with_advanced_alpha(
                                image, 
                                "JPEG", 
                                image_path, 
                                metadata=metadata, 
                                matte_color=format_params["matte_color"], 
                                save_alpha=format_params["save_alpha_separately"],
                                prompt=prompt, 
                                extra_pnginfo=extra_pnginfo,
                                **format_params
                            )
                            
                            # Add alpha path to results if saved
                            if success and alpha_path:
                                results.append(alpha_path)

                        # Standard formats (PNG, JPG, WebP)
                        else:
                            # For 8-bit standard formats
                            pil_format = "JPEG" if file_format == "jpg" else file_format.upper()
                            success = self._save_with_alpha_and_metadata(
                                image, 
                                pil_format, 
                                image_path, 
                                metadata=metadata, 
                                prompt=prompt, 
                                extra_pnginfo=extra_pnginfo,
                                **format_params
                            )
                    except Exception as e:
                        save_error = str(e)
                        print(f"[MetadataSaveImage] Error saving image: {save_error}")
                    
                    if success:
                        # Add primary format to results
                        results.append(image_path)
                        
                        # For UI display, handle based on location
                        base_output_dir = self.output_dir
                        if os.path.dirname(image_path) != base_output_dir and not os.path.dirname(image_path).startswith(os.path.join(base_output_dir, '')):
                            # Image is outside standard output dir - create a preview
                            preview_dir = os.path.join(base_output_dir, "node_previews")
                            os.makedirs(preview_dir, exist_ok=True)
                            preview_path = os.path.join(preview_dir, os.path.basename(image_path))
                            
                            try:
                                import shutil
                                shutil.copy2(image_path, preview_path)
                                ui_results.append(preview_path)
                                print(f"[DEBUG] Created preview for external path: {preview_path}")
                            except Exception as e:
                                print(f"[MetadataSaveImage] Error creating preview: {str(e)}")
                                ui_results.append(image_path)  # Fallback to original
                        else:
                            # Image is in standard output or subfolder - use as is
                            ui_results.append(image_path)
                        
                        # Try to save additional format if requested
                        additional_format = kwargs.get('additional_format', 'None')
                        if additional_format != 'None':
                            additional_path = self._save_additional_format(image, image_path, metadata=metadata, prompt=prompt, extra_pnginfo=extra_pnginfo, **kwargs)
                            if additional_path:
                                # Add to results list
                                results.append(additional_path)
                                
                                # Handle UI display for additional format too
                                if os.path.dirname(additional_path) != base_output_dir and not os.path.dirname(additional_path).startswith(os.path.join(base_output_dir, '')):
                                    # Create preview for additional format if outside standard dir
                                    add_preview_path = os.path.join(preview_dir, os.path.basename(additional_path))
                                    try:
                                        import shutil
                                        shutil.copy2(additional_path, add_preview_path)
                                        ui_results.append(add_preview_path)
                                    except Exception as e:
                                        print(f"[MetadataSaveImage] Error creating preview for additional format: {str(e)}")
                                        ui_results.append(additional_path)  # Fallback to original
                                else:
                                    ui_results.append(additional_path)
                    
                    # Write metadata if needed
                    if kwargs.get("enable_metadata", True) and metadata_targets and metadata:
                        try:
                            # Skip embedded metadata for SVG (already done) and for formats that don't support it
                            targets_to_use = [t for t in metadata_targets if t != "embedded" or file_format != "svg"]
                            if targets_to_use:
                                # Ensure basic metadata contains all essential fields for text file
                                if metadata and 'basic' not in metadata:
                                    metadata['basic'] = {}
                                    
                                # Make sure creator and copyright are in basic metadata
                                if kwargs.get("creator") and not metadata.get('basic', {}).get('creator'):
                                    metadata['basic']['creator'] = kwargs.get("creator")
                                    
                                if kwargs.get("copyright") and not metadata.get('basic', {}).get('copyright'):
                                    metadata['basic']['copyright'] = kwargs.get("copyright")
                                    
                                if kwargs.get("title") and not metadata.get('basic', {}).get('title'):
                                    metadata['basic']['title'] = kwargs.get("title")

                                # Now write metadata with targets
                                self.metadata_service.write_metadata(
                                    image_path, 
                                    metadata, 
                                    targets=targets_to_use
                                )
                        except Exception as meta_error:
                            errors.append(f"Metadata writing error: {str(meta_error)}")
                            print(f"[MetadataSaveImage] Error writing metadata: {str(meta_error)}")
                    
                    # Save workflow as JSON if requested
                    if kwargs.get("save_workflow_as_json", False) and prompt:
                        try:
                            json_path = os.path.splitext(image_path)[0] + '.json'
                            self._save_workflow_json(json_path, prompt, extra_pnginfo)
                        except Exception as json_error:
                            errors.append(f"Workflow JSON saving error: {str(json_error)}")
                            print(f"[MetadataSaveImage] Error saving workflow JSON: {str(json_error)}")
                    
                    # Create UI preview if using custom directory
                    if output_dir != base_output_dir:
                        ui_path = self._create_preview_copy(image_path, base_output_dir, filename)
                        ui_results.append(ui_path if ui_path else image_path)
                    else:
                        ui_results.append(image_path)
                except Exception as img_error:
                    errors.append(f"Image processing error: {str(img_error)}")
                    print(f"[MetadataSaveImage] Error processing image {img_index}: {str(img_error)}")
            
            # Generate workflow discovery report if enabled
            if hasattr(self.workflow_extractor, 'discovery_mode') and self.workflow_extractor.discovery_mode:
                try:
                    self._update_workflow_discovery(base_output_dir, results, **kwargs)
                except Exception as discovery_error:
                    errors.append(f"Workflow discovery error: {str(discovery_error)}")
                    print(f"[MetadataSaveImage] Error updating workflow discovery: {str(discovery_error)}")
        
        except Exception as global_error:
            errors.append(f"Global error: {str(global_error)}")
            print(f"[MetadataSaveImage] Global error in save process: {str(global_error)}")
        
        
        # Log error summary if errors occurred
        if errors and self.debug:
            print(f"[MetadataSaveImage] Completed with {len(errors)} errors:")
            for i, error in enumerate(errors):
                print(f"  {i+1}. {error}")

        # Create proper UI data for ComfyUI to display the images
        ui = self._prepare_ui_data(ui_results if ui_results else results, subfolder)
        
        if self.debug and 'images' in ui:
            print(f"[DEBUG] UI display paths for {len(ui['images'])} images:")
            for i, img in enumerate(ui['images']):
                print(f"  {i+1}. {img['subfolder']}/{img['filename']}")
        
        # Return the original images tensor, first filepath, and UI data
        return {
            'ui': ui,
            'result': (original_images, results[0] if results else "")
        }

    def _save_additional_format(self, image, primary_path, metadata, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Save a second copy of the image in a different format
        
        Args:
            image: The image data
            primary_path: Path to the primary saved image
            metadata: Metadata to embed
            prompt: Optional workflow prompt data
            extra_pnginfo: Additional PNG info
            **kwargs: Format options
            
        Returns:
            str: Path to the additional format file or empty string
        """
        # Check if additional format is requested
        additional_format = kwargs.get('additional_format', 'None')
        if additional_format == 'None':
            return ""
        
        try:
            # Create new filename with suffix
            base_path = os.path.splitext(primary_path)[0]
            suffix = kwargs.get('additional_format_suffix', '_web')
            additional_path = f"{base_path}{suffix}.{additional_format.lower()}"
            
            # Get additional format options
            quality_preset = kwargs.get('additional_format_quality', 'Balanced')
            embed_workflow = kwargs.get('additional_format_embed_workflow', False)
            color_profile_name = kwargs.get('additional_format_color_profile', 'sRGB v4 Appearance')
            
            # Prepare workflow data if needed
            workflow_pnginfo = None
            if embed_workflow and additional_format.lower() == 'png' and prompt:
                workflow_pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
            
            # Get color profile data
            color_profile_data = self._get_color_profile(color_profile_name)
            
            # Create format parameters
            format_params = {
                'quality_preset': quality_preset,
                'color_profile_name': color_profile_name,
                'color_profile_data': color_profile_data,
                'workflow_pnginfo': workflow_pnginfo,
                'png_compression': 6 if quality_preset == 'Balanced' else (1 if quality_preset == 'Best Quality' else 9),
                'jpg_quality': 90 if quality_preset == 'Balanced' else (95 if quality_preset == 'Best Quality' else 80),
                'webp_quality': 85 if quality_preset == 'Balanced' else (95 if quality_preset == 'Best Quality' else 75)
            }
            
            # Use appropriate save method based on format
            success = False
            
            if additional_format.lower() == 'png':
                # Use standard PNG saving
                pil_image = self._convert_to_pil(image)
                save_options = {
                    'compress_level': format_params['png_compression'],
                    'icc_profile': color_profile_data
                }
                pil_image.save(
                    additional_path, 
                    format='PNG', 
                    pnginfo=workflow_pnginfo,
                    **save_options
                )
                success = True
                
            elif additional_format.lower() == 'jpg':
                # Use JPG saving with appropriate quality
                pil_image = self._convert_to_pil(image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                    
                pil_image.save(
                    additional_path, 
                    format='JPEG', 
                    quality=format_params['jpg_quality'], 
                    optimize=True,
                    icc_profile=color_profile_data
                )
                success = True
                
            elif additional_format.lower() == 'webp':
                # Use WebP saving
                pil_image = self._convert_to_pil(image)
                pil_image.save(
                    additional_path, 
                    format='WEBP', 
                    quality=format_params['webp_quality'],
                    icc_profile=color_profile_data
                )
                success = True
                
            elif additional_format.lower() in ['tiff', 'tif']:
                # Use our TIFF saving method
                success = self._save_as_tiff(
                    image,
                    additional_path,
                    metadata=metadata,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    **format_params
                )
            elif additional_format.lower() == 'psd':
                # Use our PSD saving method
                success = self._save_as_psd(image, additional_path, metadata=metadata, prompt=prompt, extra_pnginfo=extra_pnginfo, **kwargs)
                
            elif additional_format.lower() == 'svg' and HAS_VTRACER:
                # Use our SVG saving method
                success = self._save_as_svg(image, additional_path, metadata=metadata, prompt=prompt, extra_pnginfo=extra_pnginfo, **kwargs)
            
            # Write metadata if needed and not already embedded
            if success and metadata and additional_format.lower() not in ['svg']:
                try:
                    targets = ["xmp"]
                    if additional_format.lower() not in ['jpg', 'jpeg']:
                        targets.append("embedded")
                        
                    self.metadata_service.write_metadata(additional_path, metadata, targets=targets)
                except Exception as meta_err:
                    print(f"[MetadataSaveImage] Error writing metadata to additional format: {str(meta_err)}")
            
            if success:
                print(f"[MetadataSaveImage] Saved additional format: {additional_path}")
                return additional_path
            else:
                print(f"[MetadataSaveImage] Failed to save additional format: {additional_format}")
                return ""
                
        except Exception as e:
            print(f"[MetadataSaveImage] Error saving additional format: {str(e)}")
            return ""


    def build_metadata_from_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Build metadata structure from node inputs with proper Adobe compatibility
        
        Args:
            **kwargs: Node inputs
            
        Returns:
            dict: Metadata structure
        """
        import uuid
        
        metadata = {
            "basic": {},                  # Our internal structure
            "dc": {},                     # Dublin Core namespace 
            "photoshop": {},              # Photoshop namespace
            "xmp": {},                    # XMP core namespace
            "xmpRights": {},              # XMP Rights Management
            "xmpMM": {},                  # XMP Media Management
            "Iptc4xmpCore": {},           # IPTC Core namespace
            "analysis": {},               # Our analysis data
            "ai_info": {                  # Our AI information
                "generation": {},
                "workflow_info": {}
            }
        }
        
        # Extract basic metadata fields
        basic = metadata["basic"]
        
        # Handle title with proper mapping for Adobe
        if kwargs.get("title"):
            title = kwargs["title"].strip()
            # Our internal structure - simple string
            basic["title"] = title
            
            # Dublin Core - proper structure for language alternatives
            metadata["dc"]["title"] = {"x-default": title}
            
            # Photoshop fields - direct string
            metadata["photoshop"]["Headline"] = title  # Important for Photoshop
            metadata["photoshop"]["DocumentTitle"] = title
            metadata["photoshop"]["Title"] = title  # Add this explicit field
            
            # IPTC Core - proper structure
            metadata["Iptc4xmpCore"]["Title"] = {"x-default": title}
        
        # Handle description
        if kwargs.get("description"):
            description = kwargs["description"].strip()
            # Our internal structure - simple string
            basic["description"] = description
            
            # Dublin Core - proper structure
            metadata["dc"]["description"] = {"x-default": description}
            
            # Photoshop fields - direct string
            metadata["photoshop"]["Caption"] = description  # Important field
            metadata["photoshop"]["Caption-Abstract"] = description  # Critical field for Lightroom
            metadata["photoshop"]["Description"] = description
            
            # IPTC Core - proper structure
            metadata["Iptc4xmpCore"]["Description"] = {"x-default": description}
        
        # Handle creator with proper mapping
        if kwargs.get("creator"):
            creator = kwargs["creator"].strip()
            # Our internal structure
            basic["creator"] = creator
            
            # Dublin Core as a proper sequence
            metadata["dc"]["creator"] = [creator]
            
            # Photoshop author fields - direct strings
            metadata["photoshop"]["Author"] = creator # Critical for Photoshop
            metadata["photoshop"]["AuthorsPosition"] = "Creator"
            metadata["photoshop"]["Credit"] = f"Image by {creator}"
            metadata["photoshop"]["Byline"] = creator  # Critical for some Adobe products
            
            # IPTC Core
            metadata["Iptc4xmpCore"]["Creator"] = [creator]
            
            # XMP fields that Photoshop also looks at
            metadata["xmp"]["Author"] = creator
            metadata["xmp"]["CreatorTool"] = "ComfyUI"  # Always include this
        
        # Handle copyright with proper fields
        if kwargs.get("copyright"):
            copyright_text = kwargs["copyright"].strip()
            # Our internal structure
            basic["copyright"] = copyright_text
            basic["rights"] = copyright_text
            
            # Dublin Core with proper structure
            metadata["dc"]["rights"] = {"x-default": copyright_text}
            
            # XMP Rights Management - critical for Photoshop
            metadata["xmpRights"]["Marked"] = True
            metadata["xmpRights"]["UsageTerms"] = {"x-default": copyright_text}
            metadata["xmpRights"]["WebStatement"] = copyright_text
            if kwargs.get("creator"):
                metadata["xmpRights"]["Owner"] = [kwargs.get("creator")]
            
            # Photoshop fields - direct strings
            metadata["photoshop"]["Copyright"] = copyright_text  # Critical field
            metadata["photoshop"]["CopyrightNotice"] = copyright_text
            metadata["photoshop"]["CopyrightStatus"] = "Copyrighted"
            
            # IPTC Core
            metadata["Iptc4xmpCore"]["CopyrightNotice"] = {"x-default": copyright_text}
        
        # Handle project in basic metadata
        if kwargs.get("project"):
            project = kwargs["project"].strip()
            # Our internal structure
            basic["project"] = project
            
            # Store in custom namespaces for compatibility
            metadata["xmp"]["Label"] = project
            
            # XMP Media Management
            metadata["xmpMM"]["DocumentID"] = f"xmp.did:project:{project.replace(' ', '_')}"
            metadata["xmpMM"]["OriginalDocumentID"] = metadata["xmpMM"]["DocumentID"]
            
            # Store in AI namespace as a backup
            metadata["ai_info"]["project"] = project
        
        # Handle keywords (comma-separated)
        if kwargs.get("keywords"):
            # Split by commas and strip whitespace
            keyword_list = [kw.strip() for kw in kwargs["keywords"].split(",") if kw.strip()]
            if keyword_list:
                # Our internal structure
                basic["keywords"] = keyword_list
                
                # Dublin Core uses subject for keywords
                metadata["dc"]["subject"] = keyword_list
                
                # IPTC Core
                metadata["Iptc4xmpCore"]["Keywords"] = keyword_list
                
                # Add to Photoshop namespace as well
                metadata["photoshop"]["Keywords"] = ", ".join(keyword_list)

        # Add text layers info if available from inputs
        text_layers = []
        
        # Check for mask layers with names
        for i in range(1, 3):  # Up to 2 masks
            mask_key = f"mask_layer_{i}"
            name_key = f"mask_{i}_name"
            if mask_key in kwargs and kwargs[mask_key] is not None and name_key in kwargs:
                text_layers.append({
                    "name": kwargs[name_key],
                    "type": "Mask"
                })
                
        # Check for overlay layers with names
        for i in range(1, 3):  # Up to 2 overlays
            overlay_key = f"overlay_layer_{i}"
            name_key = f"overlay_{i}_name"
            if overlay_key in kwargs and kwargs[overlay_key] is not None and name_key in kwargs:
                text_layers.append({
                    "name": kwargs[name_key],
                    "type": "Overlay"
                })
                
        if text_layers:
            basic["text_layers"] = text_layers
            metadata["photoshop"]["LayerNames"] = [layer["name"] for layer in text_layers]
        
        # Add CreatorTool - important for Adobe
        metadata["xmp"]["CreatorTool"] = "ComfyUI"

        # Add timestamp
        generation = metadata["ai_info"]["generation"]
        generation["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add timestamps
        now = datetime.datetime.now().isoformat()
        metadata["xmp"]["CreateDate"] = now
        metadata["xmp"]["ModifyDate"] = now
        metadata["xmp"]["MetadataDate"] = now
        metadata["ai_info"]["generation"]["timestamp"] = now

        # Add document ID for proper DAM systems
        metadata["xmpMM"]["InstanceID"] = f"xmp.iid:{uuid.uuid4()}"
        if "DocumentID" not in metadata["xmpMM"]:
            metadata["xmpMM"]["DocumentID"] = f"xmp.did:{uuid.uuid4()}"
        metadata["xmpMM"]["OriginalDocumentID"] = metadata["xmpMM"]["DocumentID"]
        
        # Add custom metadata if provided
        if kwargs.get("custom_metadata"):
            try:
                custom_data = json.loads(kwargs["custom_metadata"])
                if self.debug:
                    print(f"[MetadataSaveImage] Merging custom metadata: {json.dumps(custom_data, indent=2)}")
                self._merge_custom_metadata(metadata, custom_data)
            except Exception as e:
                print(f"[MetadataSaveImage] Error parsing custom metadata: {str(e)}")
        
        return metadata

    def _merge_custom_metadata(self, metadata: Dict[str, Any], custom_data: Dict[str, Any]) -> None:
        """
        Merge custom metadata into the main structure with Adobe compatibility
        
        Args:
            metadata: Target metadata structure to update (modified in place)
            custom_data: Custom data to merge in
        """
        # First, check for Adobe-specific namespaces in custom data
        adobe_namespaces = ['dc', 'photoshop', 'xmp', 'xmpRights', 'xmpMM', 'Iptc4xmpCore']
        
        # For each top-level category in custom data
        for category, data in custom_data.items():
            # Handle Adobe namespaces specially to maintain correct structures
            if category in adobe_namespaces:
                if category in metadata:
                    self._merge_adobe_namespace(metadata[category], data, category)
                else:
                    metadata[category] = data
            elif category == 'basic':
                # Merge basic data and ensure it propagates to Adobe namespaces
                if 'basic' in metadata:
                    self._deep_merge(metadata['basic'], data)
                    
                    # Propagate key fields to Adobe namespaces
                    self._propagate_basic_to_adobe(metadata, data)
            elif category == 'eiqa' or category == 'ai':
                # These are our custom namespaces - merge directly
                if category not in metadata:
                    metadata[category] = {}
                
                # Deep merge
                self._deep_merge(metadata[category], data)
            elif category == 'ai_info':
                # Special handling for AI generation info
                if category in metadata:
                    # Ensure generation data is properly merged
                    if 'generation' in data and 'generation' in metadata[category]:
                        self._deep_merge(metadata[category]['generation'], data['generation'])
                        
                    # Merge workflow_info
                    if 'workflow_info' in data and 'workflow_info' in metadata[category]:
                        self._deep_merge(metadata[category]['workflow_info'], data['workflow_info'])
                        
                    # Handle other ai_info sections
                    for key, value in data.items():
                        if key not in ['generation', 'workflow_info', 'workflow']:
                            metadata[category][key] = value
                            
                    # Special case: don't duplicate workflow data
                    if 'workflow' in data and 'workflow' not in metadata[category]:
                        metadata[category]['workflow'] = data['workflow']
                else:
                    metadata[category] = data
            elif category in metadata:
                # Standard category that already exists - merge
                if isinstance(metadata[category], dict) and isinstance(data, dict):
                    self._deep_merge(metadata[category], data)
                else:
                    # Replace with custom data
                    metadata[category] = data
            else:
                # New category - add directly
                metadata[category] = data

    def _merge_adobe_namespace(self, target: Dict[str, Any], source: Dict[str, Any], namespace: str) -> None:
        """
        Merge Adobe namespace data with special handling for structured fields
        
        Args:
            target: Target namespace dictionary to update
            source: Source namespace dictionary with values to merge in
        """
        # Special fields that need language alternative structures
        lang_alt_fields = {
            'dc': ['title', 'description', 'rights'],
            'Iptc4xmpCore': ['Title', 'Description', 'CopyrightNotice'],
            'xmpRights': ['UsageTerms']
        }
        
        # Fields that need sequence structures
        seq_fields = {
            'dc': ['creator', 'contributor'],
            'Iptc4xmpCore': ['Creator']
        }
        
        # Fields that need bag structures
        bag_fields = {
            'dc': ['subject'],
            'Iptc4xmpCore': ['Keywords'],
            'xmpRights': ['Owner']
        }
        
        for key, value in source.items():
            # Check if this field needs special handling
            if namespace in lang_alt_fields and key in lang_alt_fields[namespace]:
                # Language alternative structure
                if isinstance(value, dict) and 'x-default' in value:
                    # Already in correct format
                    target[key] = value
                else:
                    # Convert to proper format
                    target[key] = {'x-default': str(value)}
            elif namespace in seq_fields and key in seq_fields[namespace]:
                # Sequence structure
                if isinstance(value, list):
                    # Already in correct format
                    target[key] = value
                else:
                    # Convert to proper format
                    target[key] = [str(value)]
            elif namespace in bag_fields and key in bag_fields[namespace]:
                # Bag structure
                if isinstance(value, list):
                    # Already in correct format
                    target[key] = value
                else:
                    # Convert to proper format
                    target[key] = [str(value)]
            elif isinstance(value, dict) and isinstance(target.get(key, {}), dict):
                # Recursively merge nested dictionaries
                if key not in target:
                    target[key] = {}
                self._deep_merge(target[key], value)
            else:
                # Simple value
                target[key] = value

    def _propagate_basic_to_adobe(self, metadata: Dict[str, Any], basic_data: Dict[str, Any]) -> None:
        """
        Propagate basic metadata fields to appropriate Adobe namespace locations
        
        Args:
            metadata: Full metadata structure
            basic_data: Basic data to propagate
        """
        if 'title' in basic_data:
            title = basic_data['title']
            if 'dc' in metadata:
                metadata['dc']['title'] = {'x-default': title}
            if 'photoshop' in metadata:
                metadata['photoshop']['Headline'] = title
                metadata['photoshop']['DocumentTitle'] = title
                metadata['photoshop']['Title'] = title
            if 'Iptc4xmpCore' in metadata:
                metadata['Iptc4xmpCore']['Title'] = {'x-default': title}
                
        if 'description' in basic_data:
            desc = basic_data['description']
            if 'dc' in metadata:
                metadata['dc']['description'] = {'x-default': desc}
            if 'photoshop' in metadata:
                metadata['photoshop']['Caption'] = desc
                metadata['photoshop']['Caption-Abstract'] = desc
                metadata['photoshop']['Description'] = desc
            if 'Iptc4xmpCore' in metadata:
                metadata['Iptc4xmpCore']['Description'] = {'x-default': desc}
                
        if 'creator' in basic_data:
            creator = basic_data['creator']
            if 'dc' in metadata:
                metadata['dc']['creator'] = [creator]
            if 'photoshop' in metadata:
                metadata['photoshop']['Author'] = creator
                metadata['photoshop']['Byline'] = creator
                metadata['photoshop']['Credit'] = f"Image by {creator}"
            if 'Iptc4xmpCore' in metadata:
                metadata['Iptc4xmpCore']['Creator'] = [creator]
            if 'xmp' in metadata:
                metadata['xmp']['Author'] = creator
                
        if 'copyright' in basic_data or 'rights' in basic_data:
            copyright_text = basic_data.get('copyright', basic_data.get('rights', ''))
            if 'dc' in metadata:
                metadata['dc']['rights'] = {'x-default': copyright_text}
            if 'photoshop' in metadata:
                metadata['photoshop']['Copyright'] = copyright_text
                metadata['photoshop']['CopyrightNotice'] = copyright_text
                metadata['photoshop']['CopyrightStatus'] = "Copyrighted"
            if 'Iptc4xmpCore' in metadata:
                metadata['Iptc4xmpCore']['CopyrightNotice'] = {'x-default': copyright_text}
            if 'xmpRights' in metadata:
                metadata['xmpRights']['Marked'] = True
                metadata['xmpRights']['UsageTerms'] = {'x-default': copyright_text}

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge nested dictionaries
        
        Args:
            target: Target dictionary to update (modified in place)
            source: Source dictionary with values to merge in
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recurse if both values are dicts
                self._deep_merge(target[key], value)
            else:
                # Otherwise replace/set the value
                target[key] = value

    def _ensure_serializable_metadata(self, metadata):
        """
        Ensure metadata is fully serializable without any PIL objects
        
        Args:
            metadata: Original metadata
            
        Returns:
            dict: Cleaned metadata
        """
        if hasattr(metadata, 'chunks'):  # Detect PngInfo objects
            # If it's a PngInfo object, return an empty dict
            return {}
            
        if isinstance(metadata, dict):
            result = {}
            for key, value in metadata.items():
                # Skip non-serializable objects
                if hasattr(value, 'chunks'):  # PngInfo check
                    continue
                    
                # Recursively clean nested structures
                if isinstance(value, dict):
                    result[key] = self._ensure_serializable_metadata(value)
                elif isinstance(value, list):
                    result[key] = [self._ensure_serializable_metadata(item) if isinstance(item, (dict, list)) else item 
                                  for item in value if not hasattr(item, 'chunks')]  # Skip PngInfo in lists
                else:
                    # Try to ensure the value is serializable
                    try:
                        json.dumps({key: value})  # Test serialization
                        result[key] = value
                    except (TypeError, OverflowError, ValueError):
                        # If serialization fails, convert to string
                        try:
                            result[key] = str(value)
                        except:
                            # If string conversion fails, skip this value
                            pass
            return result
        elif isinstance(metadata, list):
            return [self._ensure_serializable_metadata(item) if isinstance(item, (dict, list)) else item 
                   for item in metadata if not hasattr(item, 'chunks')]  # Skip PngInfo in lists
        else:
            # For other types, return as is if they're serializable
            try:
                json.dumps(metadata)
                return metadata
            except (TypeError, OverflowError, ValueError):
                # If serialization fails, try string conversion
                try:
                    return str(metadata)
                except:
                    # Last resort: return None for problematic values
                    return None
    
    def resolve_output_path(self, **kwargs):
        """
        Resolve all output paths with clear logic
        
        Args:
            **kwargs: Node parameters
            
        Returns:
            dict: Path information with keys:
                - output_dir: Full path to output directory
                - filename_prefix: Processed filename prefix
                - subfolder: Relative subfolder for UI display
                - base_output_dir: Original ComfyUI output directory
        """
        # Extract parameters
        filename_prefix_input = kwargs.get("filename_prefix", "ComfyUI")
        include_datetime = kwargs.get("include_datetime", True)
        include_project = kwargs.get("include_project", False)
        project = kwargs.get("project", "")
        custom_output_directory = kwargs.get("custom_output_directory", "")
        output_path_mode = kwargs.get("output_path_mode", "Subfolder in Output")
        
        # Debug logs
        print(f"[DEBUG] Initial filename_prefix: {filename_prefix_input}")
        print(f"[DEBUG] Include datetime: {include_datetime}")
        print(f"[DEBUG] Include project: {include_project}")
        print(f"[DEBUG] Custom output directory: {custom_output_directory}")
        print(f"[DEBUG] Output path mode: {output_path_mode}")

        # Define enhanced date replacement function
        def replace_date(match):
            date_format = match.group(1)
            now = datetime.datetime.now()
            print(f"[DEBUG] Processing date format: '{date_format}'")
            
            try:
                # Handle direct format without processing first
                if date_format in ['MMdd', 'ddMM', 'yyyyMMdd', 'MMddyyyy']:
                    result = ""
                    if date_format == 'MMdd':
                        result = now.strftime("%m%d")
                    elif date_format == 'ddMM':
                        result = now.strftime("%d%m")
                    elif date_format == 'yyyyMMdd':
                        result = now.strftime("%Y%m%d")
                    elif date_format == 'MMddyyyy':
                        result = now.strftime("%m%d%Y")
                    print(f"[DEBUG] Direct date format '{date_format}' -> '{result}'")
                    return result
                    
                # Special handling for month abbreviation
                if 'MMM' in date_format:
                    # This is a custom format with month abbreviation
                    result = ""
                    i = 0
                    while i < len(date_format):
                        if date_format[i:i+3] == 'MMM':
                            # Add abbreviated month name
                            result += now.strftime("%b")
                            i += 3
                        elif date_format[i:i+2] == 'MM':
                            # Add month number with leading zero
                            result += now.strftime("%m")
                            i += 2
                        elif date_format[i:i+1] == 'M':
                            # Add month number without leading zero
                            try:
                                result += now.strftime("%-m")  # Unix-style format
                            except ValueError:
                                # Windows doesn't support '%-m', use alternative
                                result += str(now.month)
                            i += 1
                        elif date_format[i:i+2] == 'dd':
                            # Add day with leading zero
                            result += now.strftime("%d")
                            i += 2
                        elif date_format[i:i+1] == 'd':
                            # Add day without leading zero
                            try:
                                result += now.strftime("%-d")  # Unix-style format
                            except ValueError:
                                # Windows doesn't support '%-d', use alternative
                                result += str(now.day)
                            i += 1
                        elif date_format[i:i+4] == 'yyyy':
                            # Add 4-digit year
                            result += now.strftime("%Y")
                            i += 4
                        elif date_format[i:i+2] == 'yy':
                            # Add 2-digit year
                            result += now.strftime("%y")
                            i += 2
                        else:
                            # Pass through any other characters
                            result += date_format[i]
                            i += 1
                    print(f"[DEBUG] Custom date format '{date_format}' -> '{result}'")
                    return result
                else:
                    # Use standard strftime formatting
                    result = now.strftime(date_format)
                    print(f"[DEBUG] Standard date format '{date_format}' -> '{result}'")
                    return result
            except Exception as e:
                print(f"[MetadataSaveImage] Date format error: {str(e)}, format: {date_format}")
                return match.group(0)  # Return original if format is invalid
        
        # Process filename prefix with optional datetime and project
        filename_prefix = re.sub(r'%date:(.+?)%', replace_date, filename_prefix_input)

        if include_project and project:
            project_name = project.replace(" ", "_")
            filename_prefix = f"{filename_prefix}_{project_name}"
        
        if include_datetime:
            datetime_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename_prefix = f"{filename_prefix}_{datetime_suffix}"
        
        print(f"[DEBUG] Processed filename_prefix: {filename_prefix}")
        
        # Set base output directory (original ComfyUI output folder)
        base_output_dir = self.output_dir
        print(f"[DEBUG] Base output directory: {base_output_dir}")
        
        # Process custom output directory if provided
        resolved_output_dir = base_output_dir
        if custom_output_directory:
            custom_dir = custom_output_directory.strip()
            
            # Process custom directory for date strings
            custom_dir = re.sub(r'%date:(.+?)%', replace_date, custom_dir)
            print(f"[DEBUG] Resolved custom directory after date replacement: {custom_dir}")
            
            # Handle paths that start with backslash but don't have a drive letter
            if custom_dir.startswith('\\') and not re.match(r'^[a-zA-Z]:', custom_dir):
                if output_path_mode == "Subfolder in Output":
                    # Strip leading backslash to make it truly relative
                    custom_dir = custom_dir.lstrip('\\')
            
            if output_path_mode == "Absolute Path" or os.path.isabs(custom_dir):
                # Absolute path
                resolved_output_dir = custom_dir
            elif output_path_mode == "Subfolder in Output":
                # Relative to output directory
                resolved_output_dir = os.path.join(base_output_dir, custom_dir)
        
        # Normalize the resolved output directory
        resolved_output_dir = os.path.normpath(os.path.abspath(resolved_output_dir))
        print(f"[DEBUG] Resolved output directory: {resolved_output_dir}")
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(resolved_output_dir, exist_ok=True)
        except Exception as e:
            print(f"[MetadataSaveImage] Error creating directory: {str(e)}")
            # Fall back to default output directory
            resolved_output_dir = base_output_dir
            os.makedirs(resolved_output_dir, exist_ok=True)
        
        # Calculate relative subfolder for UI
        subfolder = ""
        if resolved_output_dir != base_output_dir:
            try:
                subfolder = os.path.relpath(resolved_output_dir, base_output_dir)
                if subfolder == '.':
                    subfolder = ""
            except ValueError:
                # If on different drives, use absolute path for subfolder
                subfolder = resolved_output_dir
        
        print(f"[DEBUG] Subfolder for UI: {subfolder}")
        
        return {
            "output_dir": resolved_output_dir,
            "filename_prefix": filename_prefix,
            "subfolder": subfolder,
            "base_output_dir": base_output_dir
        }

    def generate_sequential_filename(self, output_dir, filename_prefix, file_ext, 
                                     simple_format=False, counter=0):
        """
        Generate sequential filename based on format preference
        
        Args:
            output_dir: Output directory
            filename_prefix: Filename prefix
            file_ext: File extension with dot
            simple_format: Whether to use simple format (file1.png)
            counter: Starting counter
        
        Returns:
            tuple: (filename, new_counter)
        """
        print(f"[DEBUG] Generating filename in directory: {output_dir}")
        print(f"[DEBUG] Using filename prefix: {filename_prefix}")
        print(f"[DEBUG] File extension: {file_ext}")
        print(f"[DEBUG] Simple format: {simple_format}, Starting counter: {counter}")
        
        if simple_format:
            # Simple format (file1.png)
            if counter == 0:
                # Try with no number first
                filename = f"{filename_prefix}{file_ext}"
                if os.path.exists(os.path.join(output_dir, filename)):
                    # File exists, start numbering
                    counter = 1
                    while os.path.exists(os.path.join(output_dir, f"{filename_prefix}{counter}{file_ext}")):
                        counter += 1
                    filename = f"{filename_prefix}{counter}{file_ext}"
                # counter remains 0 if no number needed
            else:
                # Use provided counter
                filename = f"{filename_prefix}{counter}{file_ext}"
                # Ensure unique
                while os.path.exists(os.path.join(output_dir, filename)):
                    counter += 1
                    filename = f"{filename_prefix}{counter}{file_ext}"
        else:
            # Default ComfyUI format with padded zeros
            filename = f"{filename_prefix}_{counter:03}{file_ext}"
            # Ensure unique
            while os.path.exists(os.path.join(output_dir, filename)):
                counter += 1
                filename = f"{filename_prefix}_{counter:03}{file_ext}"
        
        print(f"[DEBUG] Generated filename: {filename}")
        return filename, counter
    
    def _get_format_parameters(self, **kwargs):
        """Get format-specific parameters based on quality preset and other settings"""
        
        # Get quality preset and set format-specific parameters
        file_format = kwargs.get("file_format", "png").lower()
        quality_preset = kwargs.get("quality_preset", "Balanced")
        
        # Map quality preset to format-specific settings
        if quality_preset == "Best Quality":
            png_compression = 1  # Lower compression for better quality
            jpg_quality = 95
            webp_quality = 95
        elif quality_preset == "Balanced":
            png_compression = 6  # Medium compression
            jpg_quality = 90
            webp_quality = 85
        else:  # "Smallest File"
            png_compression = 9  # Maximum compression
            jpg_quality = 75
            webp_quality = 70
        
        # Get alpha channel handling parameters
        alpha_mode = kwargs.get("alpha_mode", "auto")
        matte_color = kwargs.get("matte_color", "#FFFFFF")
        save_alpha_separately = kwargs.get("save_alpha_separately", False)
        
        # Get APNG parameters if needed
        apng_fps = kwargs.get("apng_fps", 10)
        apng_loops = kwargs.get("apng_loops", 0)
        
        # Return format-specific parameters
        return {
            "png_compression": png_compression,
            "jpg_quality": jpg_quality,
            "webp_quality": webp_quality,
            "color_profile_name": kwargs.get("color_profile", "sRGB v4 Appearance"),
            "workflow_pnginfo": None,  # Will be set separately if needed
            "alpha_mode": alpha_mode,
            "matte_color": matte_color,
            "save_alpha_separately": save_alpha_separately,
            "apng_fps": apng_fps,
            "apng_loops": apng_loops
        }
    
    def _save_workflow_json(self, json_path, prompt=None, extra_pnginfo=None):
        """Save workflow data as JSON file"""
        try:
            workflow_data = None
            
            # First check if we have the workflow in a format that already works
            if extra_pnginfo and "workflow" in extra_pnginfo:
                # Use the workflow directly - it should already have the right structure
                workflow_data = extra_pnginfo["workflow"]
                
                # Ensure it has a version field
                if "version" not in workflow_data:
                    workflow_data["version"] = 0.4
            # Otherwise, try to extract from the prompt
            elif prompt is not None:
                # If prompt itself IS the workflow (common case)
                if isinstance(prompt, dict) and "nodes" in prompt:
                    workflow_data = prompt
                    
                    # Ensure it has a version field
                    if "version" not in workflow_data:
                        workflow_data["version"] = 0.4
                
                # If prompt has the complete structure with version at top level
                elif isinstance(prompt, dict) and "version" in prompt:
                    workflow_data = prompt
            
            # If we found a proper workflow structure, save it directly
            if workflow_data:
                # Write JSON file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(workflow_data, f, indent=2, ensure_ascii=False)
                print(f"[MetadataSaveImage] Saved loadable workflow as JSON: {json_path}")
            else:
                # Fallback: Save the prompt directly for reference
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(prompt, f, indent=2, ensure_ascii=False)
                print(f"[MetadataSaveImage] Saved prompt data as JSON (may not be loadable)")
        except Exception as e:
            print(f"[MetadataSaveImage] Error saving workflow as JSON: {str(e)}")
    
    def _create_preview_copy(self, image_path, base_output_dir, filename):
        """Create a copy in the standard output directory for node preview"""
        try:
            # Create a dedicated preview folder
            preview_dir = os.path.join(base_output_dir, "node_previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # Full path to preview file
            preview_path = os.path.join(preview_dir, filename)
            
            # Only copy if the destination doesn't already exist or is different
            if not os.path.exists(preview_path) or os.path.getmtime(image_path) > os.path.getmtime(preview_path):
                import shutil
                # Copy the file with metadata intact
                shutil.copy2(image_path, preview_path)
                print(f"[DEBUG] Created/updated preview copy: {preview_path}")
            
            return preview_path
        except Exception as e:
            print(f"[MetadataSaveImage] Warning: Could not create preview copy: {str(e)}")
            return None
    def _handle_alpha_channel(self, image, format, matte_color=None):
        """
        Advanced alpha channel handling for formats that don't support transparency
        
        Args:
            image: PIL Image to process
            format: Target format (e.g., "JPEG")
            matte_color: Background color for alpha compositing (None=white)
            
        Returns:
            tuple: (processed_image, alpha_image) - Alpha may be None if not extracted
        """
        if image.mode != 'RGBA' or format.upper() in ('PNG', 'WEBP', 'TIFF', 'PSD', 'APNG'):
            # Format supports alpha or image doesn't have alpha
            if image.mode == 'RGBA' and format.upper() in ('PNG', 'WEBP', 'TIFF', 'PSD', 'APNG'):
                # Already in correct format with alpha support
                return image, None
            elif image.mode != 'RGBA':
                # No alpha channel to handle
                return image, None
                
        # At this point, we have an RGBA image and a format that doesn't support alpha
        
        # Extract alpha channel
        r, g, b, alpha = image.split()
        
        # Create RGB image
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Handle matting options
        if matte_color:
            try:
                # Parse color
                if isinstance(matte_color, str):
                    from PIL import ImageColor
                    bg_color = ImageColor.getcolor(matte_color, 'RGB')
                else:
                    # Assume it's already a color tuple
                    bg_color = matte_color
                    
                # Create background
                bg = Image.new('RGB', image.size, bg_color)
                
                # Composite with alpha
                rgb_image = Image.alpha_composite(bg.convert('RGBA'), image).convert('RGB')
                
            except Exception as e:
                print(f"[MetadataSaveImage] Error applying matte color: {str(e)}")
                # Fallback to white background
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=alpha)
        else:
            # Default to white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=alpha)
        
        return rgb_image, alpha
        
    def _save_with_advanced_alpha(self, image, format, image_path, metadata=None, 
                                matte_color=None, save_alpha=True, alpha_suffix="_alpha",
                                prompt=None, extra_pnginfo=None, **kwargs):
        """
        Save image with advanced alpha channel handling
        
        Args:
            image: Image to save
            format: Format to save as (e.g., "JPEG")
            image_path: Path to save image
            metadata: Metadata to embed
            matte_color: Background color for compositing (None=white)
            save_alpha: Whether to save alpha channel separately
            alpha_suffix: Suffix to add to alpha channel filename
            prompt, extra_pnginfo: For workflow embedding
            **kwargs: Additional saving options
            
        Returns:
            tuple: (success, alpha_path) - Alpha path will be None if not saved
        """
        try:
            # Convert to PIL if needed
            if not isinstance(image, Image.Image):
                pil_image = self._convert_to_pil(image)
            else:
                pil_image = image
                
            # Process alpha channel
            processed_image, alpha_channel = self._handle_alpha_channel(pil_image, format, matte_color)
            
            # Prepare save options
            save_options = {}
            
            # Format-specific options
            if format.upper() == 'JPEG' or format.upper() == 'JPG':
                save_options["quality"] = kwargs.get("jpg_quality", 95)
                save_options["optimize"] = True
                save_options["progressive"] = True
            elif format.upper() == 'WEBP':
                save_options["quality"] = kwargs.get("webp_quality", 90)
                save_options["method"] = 4  # Higher quality encoding
                
            # Get color profile data if available
            color_profile_name = kwargs.get("color_profile", "sRGB v4 Appearance") 
            color_profile_data = self._get_color_profile(color_profile_name)
            if color_profile_data:
                save_options["icc_profile"] = color_profile_data
                
            # Add workflow info if PNG and requested
            if format.upper() == 'PNG' and kwargs.get('embed_workflow', True) and prompt:
                pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
                save_options["pnginfo"] = pnginfo
            
            # Save main image
            processed_image.save(image_path, format=format, **save_options)
            
            # Save alpha channel separately if requested and available
            alpha_path = None
            if save_alpha and alpha_channel:
                # Create alpha path by modifying original path
                name, ext = os.path.splitext(image_path)
                alpha_path = f"{name}{alpha_suffix}.png"
                
                # Save alpha as grayscale PNG
                alpha_channel.save(alpha_path, format="PNG")
                print(f"[MetadataSaveImage] Saved separate alpha channel: {alpha_path}")
            
            return True, alpha_path
            
        except Exception as e:
            print(f"[MetadataSaveImage] Error in advanced alpha handling: {str(e)}")
            return False, None
    def extract_metadata_from_workflow(self, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Extract comprehensive metadata from ComfyUI's workflow data
        focusing on the most important information
        
        Args:
            prompt: ComfyUI prompt data
            extra_pnginfo: Additional PNG info
            
        Returns:
            dict: Extracted metadata with hierarchical structure
        """
        # Only extract if we have data
        if not prompt and not extra_pnginfo:
            return {}
        
        try:
            # Use workflow metadata processor to handle extraction
            workflow_data = {
                'prompt': prompt,
                'extra_pnginfo': extra_pnginfo
            }
            
            # Extract metadata using our unified workflow processor
            metadata = self.workflow_processor.process_workflow_data(prompt, extra_pnginfo)
            
            # Build additional metadata from node inputs
            node_metadata = self.build_metadata_from_inputs(**kwargs)
            
            # Merge node inputs with workflow data
            if 'basic' in node_metadata and metadata.get('ai_info', {}).get('generation', {}):
                # Copy title and description from node inputs if provided
                for field in ['title', 'description', 'creator', 'copyright', 'keywords']:
                    if field in node_metadata['basic'] and node_metadata['basic'][field]:
                        # Add to both basic and generation data
                        if 'basic' not in metadata:
                            metadata['basic'] = {}
                        metadata['basic'][field] = node_metadata['basic'][field]
            
            # Add workflow-derived info to ensure it's included
            if hasattr(self, 'workflow_extractor'):
                # Add any workflow discovery information if available
                discovery_info = getattr(self.workflow_extractor, 'discovery_data', {})
                if discovery_info and 'ai_info' in metadata:
                    if 'workflow_info' not in metadata['ai_info']:
                        metadata['ai_info']['workflow_info'] = {}
                    metadata['ai_info']['workflow_info']['discovery'] = discovery_info
            
            # NEVER include workflow in XMP - it's too large and causes issues
            # Only workflow_info (metadata about the workflow) should be included
            if 'ai_info' in metadata:
                # Force this to false regardless of the input parameter
                metadata['ai_info']['include_workflow_in_xmp'] = False
                
                # Remove any full workflow data from the metadata
                if 'generation' in metadata['ai_info']:
                    for key in ['prompt', 'workflow_data', 'workflow']:
                        metadata['ai_info']['generation'].pop(key, None)
            
            # Remove any top-level workflow data
            for key in ['workflow', 'prompt', 'workflow_data']:
                metadata.pop(key, None)
                    
            return metadata
            
        except Exception as e:
            if self.debug:
                import traceback
                print(f"[MetadataSaveImage] Error extracting metadata: {str(e)}")
                traceback.print_exc()
            return {
                'error': f"Failed to extract metadata: {str(e)}",
                'basic': {
                    'title': kwargs.get('title', 'ComfyUI Generated Image'),
                    'description': kwargs.get('description', ''),
                    'creator': kwargs.get('creator', 'ComfyUI')
                }
            }

    def _update_workflow_discovery(self, base_output_dir, results, **kwargs):
        """Update workflow discovery reports"""
        # Create/update centralized reports
        central_dir = os.path.join(base_output_dir, "metadata_discovery")
        os.makedirs(central_dir, exist_ok=True)
        central_report_path = os.path.join(central_dir, "central_workflow_discovery.json")
        
        # Try to load existing central data first
        if hasattr(self.workflow_extractor, 'load_central_discovery_data'):
            self.workflow_extractor.load_central_discovery_data(central_report_path)
        
        # Save updated central data
        if hasattr(self.workflow_extractor, 'save_central_discovery_data'):
            self.workflow_extractor.save_central_discovery_data(central_report_path)
        
        # Generate HTML from the combined data
        central_html_path = os.path.join(central_dir, "central_workflow_discovery.html")
        self.workflow_extractor.save_html_report(central_html_path)
        
        # Save individual discovery files if enabled
        if kwargs.get("save_individual_discovery", False) and results:
            # Use the directory of the first saved image
            first_saved_image = results[0]
            image_directory = os.path.dirname(first_saved_image)
            image_filename = os.path.basename(first_saved_image)
            
            # Get full base filename without extension
            image_basename = os.path.splitext(image_filename)[0]
            
            # Clean only numeric suffixes, keep date parts
            # Remove numbered suffixes (like _001) but keep date formats
            clean_basename = re.sub(r'_\d{2,5}$', '', image_basename)
            
            # Get full base path for consistent file naming
            base_path_full = os.path.join(image_directory, clean_basename)
            
            # Create discovery filenames based on the image name
            report_path = f"{base_path_full}_discovery.json"
            html_report_path = f"{base_path_full}_discovery.html"
            
            # Save individual reports
            self.workflow_extractor.save_discovery_report(report_path)
            self.workflow_extractor.save_html_report(html_report_path)
            print(f"[DEBUG] Saved discovery reports with base path: {base_path_full}")
    
    def _prepare_ui_data(self, results, subfolder):
        """
        Prepare UI data specifically for ComfyUI node display
        
        Args:
            results: List of image paths
            subfolder: Subfolder for UI display
            
        Returns:
            dict: Formatted UI data
        """
        if not results:
            return {}
        
        # Ensure no duplicates
        unique_results = []
        seen = set()
        for path in results:
            if path not in seen:
                unique_results.append(path)
                seen.add(path)
        
        # Format for ComfyUI
        images = []
        
        # For consistent display, we care about the output directory structure
        base_output_dir = self.output_dir
        preview_dir = os.path.join(base_output_dir, "node_previews")
        
        for path in unique_results:
            # Extract filename
            filename = os.path.basename(path)
            
            # Determine correct subfolder for UI display
            display_subfolder = ""
            
            # Check if the file is already in the preview directory
            if os.path.dirname(path) == preview_dir:
                # Already a preview copy, use node_previews as subfolder
                display_subfolder = "node_previews"
                print(f"[DEBUG] UI: Using existing preview at node_previews/{filename}")
            
            # Check if the file is in the standard output directory
            elif os.path.dirname(path) == base_output_dir:
                # File in base output directory, no subfolder needed
                display_subfolder = ""
                print(f"[DEBUG] UI: Using image in base output directory: {filename}")
            
            # Check if the file is in a subfolder of the standard output
            elif os.path.dirname(path).startswith(base_output_dir):
                # File in a subfolder, extract relative path
                try:
                    display_subfolder = os.path.relpath(os.path.dirname(path), base_output_dir)
                    print(f"[DEBUG] UI: Using image in subfolder: {display_subfolder}/{filename}")
                except ValueError:
                    # Fallback for path resolution issues
                    display_subfolder = subfolder
                    print(f"[DEBUG] UI: Using provided subfolder: {subfolder}/{filename}")
            
            # File in an external directory, use preview copy
            else:
                # Check if preview copy exists
                preview_path = os.path.join(preview_dir, filename)
                if os.path.exists(preview_path):
                    # Use existing preview copy
                    display_subfolder = "node_previews"
                    print(f"[DEBUG] UI: Using existing preview for external file: node_previews/{filename}")
                else:
                    # Try to create preview copy
                    try:
                        os.makedirs(preview_dir, exist_ok=True)
                        import shutil
                        shutil.copy2(path, preview_path)
                        display_subfolder = "node_previews"
                        print(f"[DEBUG] UI: Created preview for external file: node_previews/{filename}")
                    except Exception as e:
                        # Fallback to original path with absolute subfolder
                        print(f"[MetadataSaveImage] Error creating preview: {str(e)}")
                        display_subfolder = subfolder
                        print(f"[DEBUG] UI: Fallback to provided subfolder: {subfolder}/{filename}")
            
            # Add the image with determined subfolder
            images.append({
                "filename": filename,
                "subfolder": display_subfolder,
                "type": "output"
            })
        
        # Return in proper format for ComfyUI
        return {"images": images}
    
    def _convert_to_pil(self, image):
        """
        Convert various image formats to PIL Image
        
        Args:
            image: Image as tensor, numpy array, or PIL Image
            
        Returns:
            PIL.Image: Converted image
        """
        # Handle tensor
        if torch.is_tensor(image):
            image_np = image.cpu().detach().numpy()
            if image_np.ndim == 4:
                image_np = image_np[0]  # Take first image if batched
                
            if image_np.shape[0] <= 4 and image_np.ndim == 3:
                # NCHW format, convert to HWC
                image_np = np.transpose(image_np, (1, 2, 0))
                
            # Scale to 0-255 (8-bit)
            if image_np.max() <= 1.0:
                image_np = (image_np * 255.0).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
            return Image.fromarray(image_np)
        
        # Handle numpy array
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
                
            return Image.fromarray(image)
        
        # Already a PIL image
        return image
    
    def _prepare_workflow_pnginfo(self, prompt=None, extra_pnginfo=None, embed_workflow=True):
        """
        Prepare PNG info with workflow data in ComfyUI-compatible format
        
        Args:
            prompt: ComfyUI prompt data
            extra_pnginfo: Additional PNG info
            embed_workflow: Whether to include workflow data in PNG
            
        Returns:
            PngInfo: PNG info object with workflow data if embed_workflow is True, otherwise None
        """
        if not prompt or not embed_workflow:
            return None

        try:
            from PIL import PngImagePlugin
            
            # Create PngInfo object
            pnginfo = PngImagePlugin.PngInfo()
            
            # Only add workflow data if embed_workflow is True
            if embed_workflow:
                # Add prompt to PNG info - CRITICAL for ComfyUI workflow loading
                if prompt:
                    try:
                        prompt_json = json.dumps(prompt)
                        pnginfo.add_text("prompt", prompt_json)
                        if self.debug:
                            print(f"[MetadataSaveImage] Added prompt data to PNG info ({len(prompt_json)} bytes)")
                    except Exception as e:
                        print(f"[MetadataSaveImage] Error adding prompt to PNG info: {str(e)}")
                        
                # Add workflow data from extra_pnginfo - ALSO REQUIRED for loading
                if extra_pnginfo:
                    for key, value in extra_pnginfo.items():
                        try:
                            if isinstance(value, dict):
                                value_json = json.dumps(value)
                                pnginfo.add_text(key, value_json)
                            elif isinstance(value, str):
                                pnginfo.add_text(key, value)
                            else:
                                pnginfo.add_text(key, str(value))
                        except Exception as e:
                            print(f"[MetadataSaveImage] Error adding {key} from extra_pnginfo: {str(e)}")
                    if self.debug:
                        print(f"[MetadataSaveImage] Added workflow structure from extra_pnginfo")
            
            return pnginfo
        except Exception as e:
            print(f"[MetadataSaveImage] Error preparing workflow PNG info: {str(e)}")
            return None
    
    def _get_color_profile(self, profile_name):
        """
        Get a color profile from file
        
        Args:
            profile_name: Name of the color profile to retrieve
            
        Returns:
            bytes: ICC profile data or None if unavailable
        """
        if profile_name == 'None':
            return None
                
        if ICC_PROFILES[profile_name] is None:
            try:
                # Load profile from file
                if profile_name in PROFILE_FILENAMES:
                    profile_filename = PROFILE_FILENAMES[profile_name]
                    
                    # First try in the nodes/profiles directory
                    profile_path = os.path.join(os.path.dirname(__file__), "profiles", profile_filename)
                    
                    if os.path.exists(profile_path):
                        with open(profile_path, 'rb') as f:
                            ICC_PROFILES[profile_name] = f.read()
                            print(f"[MetadataSaveImage] Loaded color profile {profile_name} from {profile_path}")
                    else:
                        # Look in standard system locations as fallback
                        system_paths = [
                            # Windows paths
                            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'System32', 'spool', 'drivers', 'color', profile_filename),
                            # macOS paths
                            f"/Library/ColorSync/Profiles/{profile_filename}",
                            f"/System/Library/ColorSync/Profiles/{profile_filename}",
                            # Linux paths
                            f"/usr/share/color/icc/colord/{profile_filename}",
                            f"/usr/share/color/icc/{profile_filename}"
                        ]
                        
                        for system_path in system_paths:
                            if os.path.exists(system_path):
                                with open(system_path, 'rb') as f:
                                    ICC_PROFILES[profile_name] = f.read()
                                    print(f"[MetadataSaveImage] Loaded color profile {profile_name} from system: {system_path}")
                                    break
                        
                        # If still not found, try to create a basic profile
                        if ICC_PROFILES[profile_name] is None and profile_name.startswith('sRGB'):
                            try:
                                from PIL import ImageCms
                                ICC_PROFILES[profile_name] = ImageCms.createProfile('sRGB').tobytes()
                                print(f"[MetadataSaveImage] Created basic sRGB profile for {profile_name}")
                            except Exception as icc_err:
                                print(f"[MetadataSaveImage] Could not create sRGB profile: {str(icc_err)}")
            except Exception as e:
                print(f"[MetadataSaveImage] Error loading color profile {profile_name}: {str(e)}")
                return None
        
        return ICC_PROFILES[profile_name]
    
    def _save_as_apng(self, images, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Save images as an Animated PNG (APNG) with optional layers
        
        Args:
            images: List or batch of images to save as animation frames
            image_path: Path to save the APNG
            metadata: Metadata to embed
            **kwargs: Additional parameters including:
                - fps: Frames per second (default 10)
                - loops: Number of loops (0 = infinite, default)
                
        Returns:
            bool: Success status
        """
        try:
            # Check if we have the APNG library
            try:
                from apng import APNG, PNG
                HAS_APNG = True
            except ImportError:
                print("[MetadataSaveImage] Warning: APNG library not found. Install with 'pip install apng'")
                HAS_APNG = False
                # Try fallback to PIL's animated PNG support
                try:
                    from PIL import Image
                    # Check if PIL version supports APNG (save_all parameter)
                    HAS_PIL_APNG = hasattr(Image, 'save_all')
                except:
                    HAS_PIL_APNG = False
                    
            if not HAS_APNG and not HAS_PIL_APNG:
                print("[MetadataSaveImage] Error: No APNG support available")
                return False
                
            # Extract parameters
            fps = kwargs.get("fps", 10)
            loops = kwargs.get("loops", 0)  # 0 = infinite looping
            
            # Convert frame delay to milliseconds
            delay = int(1000 / fps)
            
            # Ensure we have multiple images
            if torch.is_tensor(images):
                # If batch dimension is present, convert to list of images
                if len(images.shape) == 4:
                    images_list = [images[i] for i in range(images.shape[0])]
                else:
                    # Single image - just wrap in list
                    images_list = [images]
            else:
                # Assume it's already a list
                images_list = images
                
            print(f"[MetadataSaveImage] Saving {len(images_list)} frames as APNG with FPS: {fps}")
            
            # Convert all images to PIL format
            pil_frames = []
            for img in images_list:
                try:
                    pil_img = self._convert_to_pil(img)
                    pil_frames.append(pil_img)
                except Exception as e:
                    print(f"[MetadataSaveImage] Error converting frame to PIL: {str(e)}")
            
            # If using dedicated APNG library
            if HAS_APNG:
                # Create new APNG
                apng = APNG()
                
                # Add each frame
                for i, frame in enumerate(pil_frames):
                    # Save frame to temporary buffer
                    temp_buffer = io.BytesIO()
                    frame.save(temp_buffer, format="PNG")
                    temp_buffer.seek(0)
                    
                    # Add to APNG with delay
                    png = PNG.from_bytes(temp_buffer.getvalue())
                    apng.append(png, delay=delay)
                
                # Set loop count (0 = infinite)
                apng.num_plays = loops
                
                # Save APNG
                apng.save(image_path)
                
                # Add metadata via regular PNG metadata if available
                if metadata and prompt:
                    try:
                        # We can still add metadata to the APNG file
                        pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
                        pil_frames[0].save(image_path, format="PNG", pnginfo=pnginfo, save_all=True, 
                                        append_images=pil_frames[1:], duration=delay, loop=loops)
                    except Exception as meta_err:
                        print(f"[MetadataSaveImage] Error adding metadata to APNG: {str(meta_err)}")
                
            # Fallback to PIL's support
            elif HAS_PIL_APNG:
                # Get first frame
                first_frame = pil_frames[0]
                
                # Get workflow pnginfo
                pnginfo = None
                if prompt:
                    pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
                
                # Save as animated PNG
                first_frame.save(
                    image_path,
                    format="PNG",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=delay,
                    loop=loops,
                    pnginfo=pnginfo
                )
            
            print(f"[MetadataSaveImage] Successfully saved APNG with {len(pil_frames)} frames")
            return True
        
        except Exception as e:
            import traceback
            print(f"[MetadataSaveImage] Error saving as APNG: {str(e)}")
            traceback.print_exc()
            return False

    def _save_with_alpha_and_metadata(self, image, format, image_path, metadata=None, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Save image with proper alpha channel handling and metadata for any format
        
        Args:
            image: Image tensor or array
            format: Format string ('PNG', 'JPEG', 'TIFF', 'PSD', etc.)
            image_path: Path to save image
            metadata: Metadata to embed
            **kwargs: Format-specific options
            
        Returns:
            bool: Success status
        """
        # Convert to PIL image first
        pil_image = self._convert_to_pil(image)
        
        # Process based on format
        format = format.upper()
        
        # Get color profile
        color_profile_name = kwargs.get("color_profile", "sRGB v4 Appearance")
        color_profile_data = self._get_color_profile(color_profile_name)
        
        # Handle alpha channel for formats that don't support it
        alpha_channel = None
        if pil_image.mode == 'RGBA' and format in ('JPEG', 'JPG'):
            # Extract alpha channel
            alpha_channel = pil_image.split()[3]
            
            # Convert to RGB (remove alpha)
            pil_image = pil_image.convert('RGB')
            
            # Save alpha channel separately if needed
            if alpha_channel:
                alpha_path = os.path.splitext(image_path)[0] + '_alpha.png'
                alpha_channel.save(alpha_path)
                print(f"[MetadataSaveImage] Saved alpha channel separately: {alpha_path}")
        
        # Prepare save options based on format
        save_options = {}
        
        if format in ('PNG', 'TIFF', 'PSD', 'WEBP'):
            # Add ICC profile for formats that support it
            if color_profile_data:
                save_options["icc_profile"] = color_profile_data
        
        # Format-specific options
        if format == 'PNG':
            save_options["compress_level"] = kwargs.get("png_compression", 6)
            
            # Add workflow info for PNG
            workflow_pnginfo = None
            if kwargs.get('embed_workflow', True) and prompt:
                workflow_pnginfo = self._prepare_workflow_pnginfo(prompt, extra_pnginfo)
            
            # Set pnginfo
            pnginfo = workflow_pnginfo
            
        elif format in ('JPEG', 'JPG'):
            save_options["quality"] = kwargs.get("jpg_quality", 90)
            save_options["optimize"] = True
            save_options["progressive"] = True
            
        elif format == 'WEBP':
            save_options["quality"] = kwargs.get("webp_quality", 90)
            save_options["method"] = 4  # Higher quality encoding
            
        elif format == 'TIFF':
            compression = kwargs.get("quality_preset", "Balanced")
            if compression == "Best Quality":
                save_options["compression"] = "lzw"
            elif compression == "Balanced":
                save_options["compression"] = "zip"
            else:  # "Smallest File"
                save_options["compression"] = "deflate"
                
        # Save the image
        try:
            if format == 'PNG' and pnginfo:
                # Extract icc_profile from save_options to pass directly
                icc_profile = save_options.pop("icc_profile", None)
                print(f"[MetadataSaveImage] DEBUG: ICC profile present: {icc_profile is not None}")
                pil_image.save(image_path, format=format, pnginfo=pnginfo, icc_profile=icc_profile, **save_options)
            else:
                pil_image.save(image_path, format=format, **save_options)
                 
            # Handle metadata for formats that need it separately
            if metadata and format not in ('PNG'):  # PNG metadata handled above
                try:
                    # Write XMP sidecar for most formats
                    from ..handlers.xmp import XMPSidecarHandler
                    
                    xmp_handler = XMPSidecarHandler(debug=self.debug)
                    xmp_handler.write_metadata(image_path, metadata)
                    
                    print(f"[MetadataSaveImage] Added metadata via XMP sidecar for {format}")
                except Exception as meta_err:
                    print(f"[MetadataSaveImage] Error adding metadata: {str(meta_err)}")
                    
            return True
        except Exception as e:
            print(f"[MetadataSaveImage] Error saving {format} image: {str(e)}")
            return False
    
    def blend_layers(self, background, foreground, mode="normal"):
        """
        Blend foreground layer onto background with specified blend mode
        
        Args:
            background: PIL Image background
            foreground: PIL Image foreground (with alpha)
            mode: Blending mode (normal, multiply, screen, overlay, etc.)
            
        Returns:
            PIL.Image: Blended image
        """
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')
            
        # Ensure images are the same size
        if background.size != foreground.size:
            foreground = foreground.resize(background.size, Image.LANCZOS)
            
        # Convert to numpy arrays for easier manipulation
        bg = np.array(background).astype(np.float32) / 255
        fg = np.array(foreground).astype(np.float32) / 255
        
        # Extract alpha channels
        bg_alpha = bg[..., 3:4]
        fg_alpha = fg[..., 3:4]
        
        # RGB channels
        bg_rgb = bg[..., :3]
        fg_rgb = fg[..., :3]
        
        # Final alpha calculation (using Porter-Duff "over" operation)
        out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
        
        # Apply different blend modes
        if mode == "normal":
            # Standard alpha composition
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            out_rgb = out_rgb / np.maximum(out_alpha, 1e-8)  # Prevent division by zero
            
        elif mode == "multiply":
            # Multiply blend mode
            blended = bg_rgb * fg_rgb
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply multiply only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
            
        elif mode == "screen":
            # Screen blend mode
            blended = 1 - (1 - bg_rgb) * (1 - fg_rgb)
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply screen only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
            
        elif mode == "overlay":
            # Overlay blend mode
            # Conditionally blend based on background brightness
            blended = np.zeros_like(bg_rgb)
            
            # Where background is light (> 0.5)
            light_mask = bg_rgb > 0.5
            blended[light_mask] = 1 - 2 * (1 - bg_rgb[light_mask]) * (1 - fg_rgb[light_mask])
            
            # Where background is dark (<= 0.5)
            dark_mask = ~light_mask
            blended[dark_mask] = 2 * bg_rgb[dark_mask] * fg_rgb[dark_mask]
            
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply overlay only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
        
        elif mode == "add" or mode == "additive":
            # Additive blend mode (adds colors together)
            blended = np.minimum(bg_rgb + fg_rgb, 1.0)  # Cap at 1.0
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply additive only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
        
        elif mode == "subtract":
            # Subtractive blend mode (subtracts foreground from background)
            blended = np.maximum(bg_rgb - fg_rgb, 0.0)  # Floor at 0.0
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply subtractive only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
        
        elif mode == "difference":
            # Difference blend mode (absolute difference between colors)
            blended = np.abs(bg_rgb - fg_rgb)
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply difference only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
            
        elif mode == "darken":
            # Darken blend mode (takes the darker of each channel)
            blended = np.minimum(bg_rgb, fg_rgb)
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply darken only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
            
        elif mode == "lighten":
            # Lighten blend mode (takes the lighter of each channel)
            blended = np.maximum(bg_rgb, fg_rgb)
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply lighten only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
        
        elif mode == "color_dodge":
            # Color dodge blend mode
            # Brightens the background based on the foreground
            blended = np.ones_like(bg_rgb)
            # Avoid division by zero by using a small epsilon
            epsilon = 1e-8
            mask = fg_rgb < (1.0 - epsilon)
            blended[mask] = np.minimum(bg_rgb[mask] / (1.0 - fg_rgb[mask]), 1.0)
            
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply color dodge only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
        
        elif mode == "color_burn":
            # Color burn blend mode
            # Darkens the background based on the foreground
            blended = np.zeros_like(bg_rgb)
            # Avoid division by zero
            epsilon = 1e-8
            mask = fg_rgb > epsilon
            blended[mask] = 1.0 - np.minimum((1.0 - bg_rgb[mask]) / fg_rgb[mask], 1.0)
            
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            # Apply color burn only where foreground has alpha
            mask = fg_alpha > 0
            out_rgb[mask] = blended[mask]
                
        else:
            # Default to normal blending for unknown modes
            out_rgb = fg_rgb * fg_alpha + bg_rgb * bg_alpha * (1 - fg_alpha)
            out_rgb = out_rgb / np.maximum(out_alpha, 1e-8)
        
        # Create output array
        out = np.zeros_like(bg)
        out[..., :3] = np.clip(out_rgb, 0, 1)
        out[..., 3:4] = np.clip(out_alpha, 0, 1)
        
        # Convert back to 8-bit and create PIL image
        out_8bit = (out * 255).astype(np.uint8)
        result = Image.fromarray(out_8bit)
        
        return result

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'metadata_service'):
            self.metadata_service.cleanup()

    def __del__(self):
        """Clean up on deletion"""
        self.cleanup()

# Node registration
NODE_CLASS_MAPPINGS  = {
    "MetadataAwareSaveImage_v099": MetadataAwareSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetadataAwareSaveImage_v099": "Save Image with Metadata v099"
}