"""
ComfyUI Node: Eric's Image Sorter V13
Description: Streamlined image organization node that focuses on sorting images based on quality issues
    (blur, noise, grain). This version integrates with the new MetadataService system for
    improved metadata handling.
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
- glob: BSD 3-Clause
- shutil: BSD 3-Clause


Eric's Image Sorter v13 - Updated March 6, 2025

Streamlined image organization node that focuses on sorting images based on quality issues
(blur, noise, grain). This version integrates with the new MetadataService system for
improved metadata handling.
"""

import torch
import os
import shutil
import glob

# Import the metadata service from the package
from Metadata_system import MetadataService

class Image_Sorter_Node_V13:
    """Streamlined image sorter focused on quality-based organization with MetadataService integration"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_filepath": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": "sorted_images"}),
                "action": (["move", "copy", "save_new"], {
                    "default": "move",
                    "description": "Action to take with sorted images"
                }),
            },
            "optional": {
                # Core quality flags - focused on what this node does best
                "is_blurry": ("BOOLEAN", {"default": False}),
                "has_digital_noise": ("BOOLEAN", {"default": False}), 
                "has_film_grain": ("BOOLEAN", {"default": False}),
                "has_jpeg_artifacts": ("BOOLEAN", {"default": False}),
                
                # Key metrics that determine sorting
                "blur_score": ("FLOAT", {"default": None}),
                "noise_level": ("FLOAT", {"default": None}),
                "grain_level": ("FLOAT", {"default": None}),
                
                # Destination folders
                "blur_folder": ("STRING", {"default": "blurry"}),
                "noise_folder": ("STRING", {"default": "noisy"}),
                "grain_folder": ("STRING", {"default": "film_grain"}),
                "mixed_folder": ("STRING", {"default": "mixed_issues"}),
                "sharp_folder": ("STRING", {"default": "sharp"}),
                
                # Metadata options
                "write_to_xmp": ("BOOLEAN", {"default": True}),
                "embed_metadata": ("BOOLEAN", {"default": True}),
                "write_text_file": ("BOOLEAN", {"default": False}),
                "write_to_database": ("BOOLEAN", {"default": False}),
                "debug_logging": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "destination_paths", "sort_status")
    FUNCTION = "sort_images"
    CATEGORY = "Eric's Nodes/Organization"

    def __init__(self):
        """Initialize metadata service"""
        self.metadata_service = MetadataService(debug=True)

    def sort_images(self, images, input_filepath="", output_dir="sorted_images", 
                   action="move", is_blurry=False, has_digital_noise=False, 
                   has_film_grain=False, has_jpeg_artifacts=False,
                   blur_score=None, noise_level=None, grain_level=None,
                   blur_folder="blurry", noise_folder="noisy", 
                   grain_folder="film_grain", mixed_folder="mixed_issues", 
                   sharp_folder="sharp", 
                   write_to_xmp=True, embed_metadata=True, write_text_file=False,
                   write_to_database=False, debug_logging=False):
        """Sort images based on quality issues with enhanced metadata handling"""
        try:
            # Enable debug logging if requested
            if debug_logging:
                self.metadata_service.debug = True
                print("[ImageSorter] Starting image sort")
                print(f"[ImageSorter] Input filepath: {input_filepath}")
            
            # Collect quality data in a single dict for processing
            quality_data = {
                'is_blurry': is_blurry,
                'has_digital_noise': has_digital_noise,
                'has_film_grain': has_film_grain,
                'has_jpeg_artifacts': has_jpeg_artifacts,
                'blur_score': blur_score,
                'noise_level': noise_level,
                'grain_level': grain_level
            }
            
            # Determine destination using quality issues
            quality_issues = self._get_quality_issues(quality_data)
            dest_folder, status = self._determine_destination(
                quality_issues,
                {
                    'blurry': blur_folder,
                    'noisy': noise_folder,
                    'film_grain': grain_folder,
                    'mixed': mixed_folder,
                    'sharp': sharp_folder
                }
            )

            dest_path = os.path.join(output_dir, dest_folder, os.path.basename(input_filepath))
            if debug_logging:
                print(f"[ImageSorter] Status determined: {status}")
                print(f"[ImageSorter] Destination folder: {dest_folder}")

            # Handle file operations (move/copy/save_new)
            final_dest_path = self._handle_file_operations(action, input_filepath, dest_path, status)
            if debug_logging:
                print(f"[ImageSorter] File operation completed: {final_dest_path}")

            # Build minimal analysis data structure focused on sorting info
            analysis_data = self._build_analysis_data(quality_data, status)
            
            # Write metadata if filepath provided
            if input_filepath and final_dest_path:
                # Set targets based on user preferences
                targets = []
                if write_to_xmp: targets.append('xmp')
                if embed_metadata: targets.append('embedded')
                if write_text_file: targets.append('txt')
                if write_to_database: targets.append('db')
                
                if targets:
                    # Set resource identifier (important for proper XMP handling)
                    filename = os.path.basename(final_dest_path)
                    resource_uri = f"file:///{filename}"
                    self.metadata_service.set_resource_identifier(resource_uri)
                    
                    # Write metadata
                    write_results = self.metadata_service.write_metadata(final_dest_path, analysis_data, targets=targets)
                    
                    # Log results
                    if debug_logging:
                        success_targets = [t for t, success in write_results.items() if success]
                        if success_targets:
                            print(f"[ImageSorter] Successfully wrote metadata to: {', '.join(success_targets)}")
                        else:
                            print("[ImageSorter] Failed to write metadata to any target")

            return images, [final_dest_path], [status]

        except Exception as e:
            import traceback
            error_msg = f"Image sorting failed: {str(e)}"
            print(f"[ImageSorter] ERROR: {error_msg}")
            traceback.print_exc()
            return images, [input_filepath if input_filepath else "error"], ["error"]
        finally:
            # Ensure cleanup always happens
            self.cleanup()

    # Keep existing file operation handling method
    def _handle_file_operations(self, action: str, input_filepath: str, dest_path: str, status: str = None) -> str:
        """
        Handle file operations for main file and companion files (all files with same base name).
        
        Args:
            action: "move", "copy", or "save_new"
            input_filepath: Source file path
            dest_path: Initial destination path
            status: Status string for save_new naming
            
        Returns:
            str: Final destination path
        """
        try:
            # Skip file operations if no input filepath
            if not input_filepath or not os.path.exists(input_filepath):
                print(f"[ImageSorter] WARNING: Invalid input filepath: {input_filepath}")
                return dest_path
                
            final_dest = dest_path
            
            # Handle save_new renaming
            if action == "save_new" and status:
                base, ext = os.path.splitext(os.path.basename(input_filepath))
                final_dest = os.path.join(os.path.dirname(dest_path), f"{base}_{status}{ext}")
            
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(final_dest), exist_ok=True)
            
            # Function to handle a single file operation
            def handle_file(src, dst):
                try:
                    if action == "move":
                        shutil.move(src, dst)
                    elif action == "copy" or action == "save_new":
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[ImageSorter] ERROR: Error during file operation: {str(e)}")
                    return False
                return True
            
            # Handle main file
            main_success = handle_file(input_filepath, final_dest)
            if not main_success:
                print("[ImageSorter] ERROR: Main file operation failed")
                return input_filepath  # Return original path if main file fails
            
            # Handle companion files
            input_base, input_ext = os.path.splitext(input_filepath)
            directory = os.path.dirname(input_filepath)
            
            # Construct the glob pattern
            glob_pattern = os.path.join(directory, os.path.basename(input_base) + ".*")
            
            # Use glob to find all matching files
            for companion_path in glob.glob(glob_pattern):
                # Skip the input file itself
                if companion_path == input_filepath:
                    continue
                    
                final_base, _ = os.path.splitext(final_dest)
                _, companion_ext = os.path.splitext(companion_path)
                companion_dest = final_base + companion_ext
                
                # Only attempt to handle if the source exists
                if os.path.exists(companion_path):
                    if not handle_file(companion_path, companion_dest):
                        print(f"[ImageSorter] WARNING: Failed to handle companion file: {companion_path}")
            
            return final_dest
            
        except Exception as e:
            print(f"[ImageSorter] ERROR: File operation failed: {str(e)}")
            raise

    def _get_quality_issues(self, quality_data):
        """Extract quality issues from available flags and metrics"""
        issues = []
        
        # Check explicit boolean flags
        if quality_data.get('is_blurry'):
            issues.append('blurry')
        if quality_data.get('has_digital_noise'):
            issues.append('noisy')
        if quality_data.get('has_film_grain'):
            issues.append('film_grain')
        if quality_data.get('has_jpeg_artifacts'):
            issues.append('jpeg')
            
        # Check metric values with thresholds if no flags set
        if not issues:
            # Blur detection from blur_score
            if quality_data.get('blur_score') is not None:
                blur_score = float(quality_data['blur_score'])
                # Lower values typically indicate more blur
                if blur_score < 100:  # Adjust threshold as needed
                    issues.append('blurry')
                    
            # Noise detection from noise_level
            if quality_data.get('noise_level') is not None:
                noise_level = float(quality_data['noise_level'])
                if noise_level > 0.1:  # Adjust threshold as needed
                    issues.append('noisy')
                    
            # Grain detection from grain_level
            if quality_data.get('grain_level') is not None:
                grain_level = float(quality_data['grain_level'])
                if grain_level > 0.15:  # Adjust threshold as needed
                    issues.append('film_grain')
        
        return issues

    def _determine_destination(self, quality_issues, folders):
        """Determine destination folder and status"""
        if 'blurry' in quality_issues:
            # Prioritize blur as the most critical issue
            return folders['blurry'], 'blurry'
        elif len(quality_issues) > 1:
            # Multiple issues detected
            return folders['mixed'], '_'.join(sorted(quality_issues))
        elif quality_issues:
            # Single issue detected
            issue = quality_issues[0]
            return folders[issue], issue
        else:
            # No issues - sharp/clean image
            return folders['sharp'], 'sharp'

    def _build_analysis_data(self, quality_data, status):
        """Build analysis data structure with proper hierarchy"""
        # Extract all quality metrics
        quality_metrics = {
            'blur_score': quality_data.get('blur_score'),
            'noise_level': quality_data.get('noise_level'),
            'grain_level': quality_data.get('grain_level'),
            'quality_score': quality_data.get('quality_score'),
            'technical_score': quality_data.get('technical_score'),
            'aesthetic_score': quality_data.get('aesthetic_score'),
            'nima_score': quality_data.get('nima_score'),
            'clip_score': quality_data.get('clip_score')
        }
        
        # Filter out None values
        quality_metrics = {k: v for k, v in quality_metrics.items() if v is not None}
        
        # Build analysis structure
        analysis = {}
        
        # Technical metrics section
        technical = {}
        for metric in ['blur_score', 'noise_level', 'grain_level', 'technical_score']:
            if metric in quality_metrics:
                technical[metric] = {
                    'score': round(float(quality_metrics[metric]), 3),  # Round to 3 decimal places
                    'timestamp': self.metadata_service.get_timestamp()
                }
        
        # Add quality flags
        for flag in ['is_blurry', 'has_digital_noise', 'has_film_grain', 'has_jpeg_artifacts']:
            if flag in quality_data and quality_data[flag] is not None:
                technical[flag] = bool(quality_data[flag])
        
        # Aesthetic metrics section
        aesthetic = {}
        for metric in ['aesthetic_score', 'nima_score', 'clip_score', 'quality_score']:
            if metric in quality_metrics:
                aesthetic[metric] = {
                    'score': round(float(quality_metrics[metric]), 2),  # Round to 2 decimal places for aesthetic scores
                    'timestamp': self.metadata_service.get_timestamp()
                }
        
        # Add sorting info section
        sorting_info = {
            'status': status,
            'timestamp': self.metadata_service.get_timestamp()
        }
        
        # Build the final analysis structure
        if technical:
            analysis['technical'] = technical
        if aesthetic:
            analysis['aesthetic'] = aesthetic
        
        # Add quality metrics directly to classification
        classification = {
            'sort_status': status
        }
        
        # Include final result in analysis metadata
        metadata = {
            'analysis': analysis,
            'classification': classification,
            'sort_info': sorting_info
        }
        
        return metadata

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'metadata_service'):
            self.metadata_service.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

NODE_CLASS_MAPPINGS = {
    "Eric_Image_Sorter_V13": Image_Sorter_Node_V13
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eric_Image_Sorter_V13": "Eric's Image Sorter V13"
}