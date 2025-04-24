"""
ComfyUI Node: Eric's Multi-Keyword Sorter V6
Description: Advanced image sorting node that organizes images into folders based on multiple
keyword matches with integrated MetadataService support.
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

Eric's Multi-Keyword Sorter V6 - Updated March 6, 2025

Advanced image sorting node that organizes images into folders based on multiple
keyword matches with integrated MetadataService support.

Features:
- Multiple keyword/folder pair support
- Different matching modes (first, all, combined)
- Handles various file operations (move, copy, save_new)
- Preserves metadata with proper merge handling
- Supports multiple metadata storage formats

"""

import cv2
import torch
import numpy as np
import os
import shutil
import json
from datetime import datetime

# Import the metadata service from the package
from Metadata_system import MetadataService

class Keyword_Sorter_Node_V6:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_filepath": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": "sorted_images"}),
                "keyword1": ("STRING", {"default": "black and white, b&w, monochrome", "multiline": True}),
                "folder1": ("STRING", {"default": "black_and_white"}),
                "active1": ("BOOLEAN", {"default": True}),
                "keyword2": ("STRING", {"default": "portrait, headshot, face", "multiline": True}),
                "folder2": ("STRING", {"default": "portraits"}),
                "active2": ("BOOLEAN", {"default": False}),
                "keyword3": ("STRING", {"default": "", "multiline": True}),
                "folder3": ("STRING", {"default": ""}),
                "active3": ("BOOLEAN", {"default": False}),
                "keyword4": ("STRING", {"default": "", "multiline": True}),
                "folder4": ("STRING", {"default": ""}),
                "active4": ("BOOLEAN", {"default": False}),
                "keyword5": ("STRING", {"default": "", "multiline": True}),
                "folder5": ("STRING", {"default": ""}),
                "active5": ("BOOLEAN", {"default": False}),
                "keyword6": ("STRING", {"default": "", "multiline": True}),
                "folder6": ("STRING", {"default": ""}),
                "active6": ("BOOLEAN", {"default": False}),
                "action": (["move", "copy", "save_new"], {"default": "move"}),
                "multi_match": (["first", "all", "combined"], {"default": "first", 
                               "tooltip": "first: only use first match, all: process each match separately, combined: create combined folders"}),
                "match_fields": (["keywords_only", "all_fields"], {"default": "keywords_only",
                                "tooltip": "keywords_only: match only in keywords, all_fields: search all metadata fields"})
            },
            "optional": {
                # Metadata options
                "write_to_xmp": ("BOOLEAN", {"default": True, 
                    "tooltip": "Write metadata to XMP sidecar file"}),
                "embed_metadata": ("BOOLEAN", {"default": True,
                    "tooltip": "Embed metadata in the image file"}),
                "write_text_file": ("BOOLEAN", {"default": False,
                    "tooltip": "Save sorting results to text file"}),
                "write_to_database": ("BOOLEAN", {"default": False,
                    "tooltip": "Write metadata to database"}),
                "debug_logging": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable detailed debug logging"}),
                "tagger_input": ("TAGGER_OUTPUT",),
                "additional_keywords": ("STRING", {"default": "", "multiline": True, 
                                      "tooltip": "Additional keywords to include in metadata"}),
                "ignore_keywords": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "Comma-separated list of keywords to ignore during matching"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "destination_paths", "matched_keywords")
    FUNCTION = "sort_images"
    CATEGORY = "Eric's Nodes/Organization"

    def __init__(self):
        """Initialize with metadata service"""
        self.metadata_service = MetadataService(debug=True)

    def sort_images(self, images, input_filepath, output_dir, 
                   keyword1="", folder1="", active1=True,
                   keyword2="", folder2="", active2=False,
                   keyword3="", folder3="", active3=False,
                   keyword4="", folder4="", active4=False,
                   keyword5="", folder5="", active5=False,
                   keyword6="", folder6="", active6=False,
                   action="move", multi_match="first", match_fields="keywords_only",
                   write_to_xmp=True, embed_metadata=True, write_text_file=False,
                   write_to_database=False, debug_logging=False,
                   tagger_input=None, additional_keywords="", ignore_keywords=""):
        """Sort images based on keyword matching with integrated metadata handling"""
        try:
            # Enable debug logging if requested
            if debug_logging:
                self.metadata_service.debug = True
                print("[KeywordSorter] Starting keyword sort operation")

            # Validate input path
            if not input_filepath or not os.path.exists(input_filepath):
                print("[KeywordSorter] Invalid input path")
                return (images, "", "")

            # Read metadata using service
            metadata = self.metadata_service.read_metadata(input_filepath, fallback=True)
            if debug_logging:
                print(f"[KeywordSorter] Read metadata from {input_filepath}")

            # Gather keywords from all sources
            all_keywords = self._gather_keywords(
                metadata, tagger_input, additional_keywords, ignore_keywords
            )
            if debug_logging:
                print(f"[KeywordSorter] Gathered {len(all_keywords)} keywords")

            # Get active keyword/folder pairs
            keyword_folders = self._get_active_pairs([
                (keyword1, folder1, active1),
                (keyword2, folder2, active2),
                (keyword3, folder3, active3),
                (keyword4, folder4, active4),
                (keyword5, folder5, active5),
                (keyword6, folder6, active6)
            ])
            if debug_logging:
                print(f"[KeywordSorter] Active keyword/folder pairs: {len(keyword_folders)}")

            # Find matches
            matches = self._find_matches(keyword_folders, metadata, match_fields)
            if not matches:
                print("[KeywordSorter] No matches found")
                return (images, "", "")

            if debug_logging:
                print(f"[KeywordSorter] Found matches: {matches}")

            # Process matches based on mode
            if multi_match == "combined" and len(matches) > 1:
                return self._process_combined_match(
                    images, input_filepath, output_dir, matches, 
                    all_keywords, action, embed_metadata, 
                    write_to_xmp, write_text_file, write_to_database,
                    debug_logging
                )

            # Process single matches with proper error handling
            return self._process_single_matches(
                images, input_filepath, output_dir,
                matches[:1] if multi_match == "first" else matches,
                all_keywords, action, embed_metadata,
                write_to_xmp, write_text_file, write_to_database,
                debug_logging
            )

        except Exception as e:
            import traceback
            error_msg = f"Keyword sorting failed: {str(e)}"
            print(f"[KeywordSorter] ERROR: {error_msg}")
            traceback.print_exc()
            return (images, "", "")
        finally:
            # Ensure cleanup always happens
            self.cleanup()

    def _gather_keywords(self, metadata, tagger_input, additional_keywords, ignore_keywords):
        """Gather keywords from all sources"""
        try:
            all_keywords = set()

            # Process tagger input
            if tagger_input:
                if isinstance(tagger_input, (list, tuple)) and tagger_input:
                    all_keywords.update(self._process_keywords(tagger_input[0]))

            # Add additional keywords
            if additional_keywords:
                all_keywords.update(self._process_keywords(additional_keywords))

            # Add existing keywords from metadata
            if 'basic' in metadata and 'keywords' in metadata['basic']:
                existing_keywords = metadata['basic']['keywords']
                if isinstance(existing_keywords, list):
                    all_keywords.update(existing_keywords)
                elif isinstance(existing_keywords, str):
                    all_keywords.update(self._process_keywords(existing_keywords))

            # Remove ignored keywords
            if ignore_keywords:
                ignore_set = self._process_keywords(ignore_keywords)
                all_keywords -= ignore_set

            return all_keywords

        except Exception as e:
            print(f"[KeywordSorter] Keyword gathering error: {str(e)}")
            return set()

    def _process_keywords(self, keyword_string):
        """Process keyword string into a set of keywords"""
        if not keyword_string:
            return set()
            
        # Handle list input
        if isinstance(keyword_string, (list, tuple, set)):
            keywords = set()
            for kw in keyword_string:
                keywords.update(self._process_keywords(kw))
            return keywords
            
        # Handle string input
        return {kw.strip().lower() for kw in keyword_string.split(',') if kw.strip()}

    def _find_matches(self, keyword_folders, metadata, match_fields):
        """Find keyword matches"""
        try:
            matches = []
            for keyword, folder in keyword_folders:
                if self._check_keyword_match(keyword, metadata, match_fields):
                    matches.append((keyword, folder))
            return matches

        except Exception as e:
            print(f"[KeywordSorter] Match finding error: {str(e)}")
            return []

    def _get_active_pairs(self, pairs):
        """Get active keyword/folder pairs"""
        return [(kw, folder) for kw, folder, active in pairs 
                if active and kw.strip() and folder.strip()]

    def _check_keyword_match(self, keyword_group, metadata, match_fields):
        """Check keyword matches"""
        try:
            keywords = self._process_keywords(keyword_group)
            
            # Check in keywords field first
            if 'basic' in metadata and 'keywords' in metadata['basic']:
                metadata_keywords = metadata['basic']['keywords']
                if isinstance(metadata_keywords, list):
                    metadata_keywords = {kw.lower() for kw in metadata_keywords}
                    if metadata_keywords & keywords:
                        return True
                elif isinstance(metadata_keywords, str):
                    metadata_keywords = self._process_keywords(metadata_keywords)
                    if metadata_keywords & keywords:
                        return True

            # Check other fields if requested
            if match_fields == "all_fields":
                # Check in AI info (generation data)
                if 'ai_info' in metadata and 'generation' in metadata['ai_info']:
                    gen_data = metadata['ai_info']['generation']
                    text_fields = ['prompt', 'negative_prompt', 'model']
                    
                    for field in text_fields:
                        if field in gen_data:
                            field_value = str(gen_data[field]).lower()
                            if any(kw.lower() in field_value for kw in keywords):
                                return True
                
                # Check in basic data
                if 'basic' in metadata:
                    basic_data = metadata['basic']
                    text_fields = ['title', 'description', 'caption', 'author']
                    
                    for field in text_fields:
                        if field in basic_data:
                            field_value = str(basic_data[field]).lower()
                            if any(kw.lower() in field_value for kw in keywords):
                                return True

            return False

        except Exception as e:
            print(f"[KeywordSorter] Keyword match error: {str(e)}")
            return False

    def _process_combined_match(self, images, input_filepath, output_dir, 
                              matches, keywords, action, embed_metadata, 
                              write_to_xmp, write_text_file, write_to_database,
                              debug_logging):
        """Process combined matches with metadata handling"""
        try:
            combined_keywords = "_".join(kw.replace(" ", "_") for kw, _ in matches)
            combined_folder = "_".join(folder for _, folder in matches)
            folder_path = os.path.join(output_dir, combined_folder)
            os.makedirs(folder_path, exist_ok=True)
            
            dest_path = os.path.join(folder_path, os.path.basename(input_filepath))

            # Perform file operation
            if not self._perform_file_operation(
                images, input_filepath, dest_path, action
            ):
                return (images, "", "")

            # Create metadata for combined match
            metadata = {
                'basic': {
                    'keywords': list(keywords)
                },
                'sort_info': {
                    'sort_method': 'combined',
                    'matched_keywords': [combined_keywords],
                    'sort_date': datetime.now().isoformat()
                }
            }

            # Set targets based on user preferences
            targets = []
            if write_to_xmp: targets.append('xmp')
            if embed_metadata: targets.append('embedded')
            if write_text_file: targets.append('txt')
            if write_to_database: targets.append('db')
            
            if targets:
                # Set resource identifier (important for proper XMP handling)
                filename = os.path.basename(dest_path)
                resource_uri = f"file:///{filename}"
                self.metadata_service.set_resource_identifier(resource_uri)
                
                # Write metadata
                write_results = self.metadata_service.write_metadata(dest_path, metadata, targets=targets)
                
                # Log results
                if debug_logging:
                    success_targets = [t for t, success in write_results.items() if success]
                    if success_targets:
                        print(f"[KeywordSorter] Successfully wrote metadata to: {', '.join(success_targets)}")
                    else:
                        print("[KeywordSorter] Failed to write metadata to any target")

            return (images, dest_path, combined_keywords)

        except Exception as e:
            print(f"[KeywordSorter] Combined match processing error: {str(e)}")
            return (images, "", "")

    def _process_single_matches(self, images, input_filepath, output_dir,
                              matches, keywords, action, embed_metadata,
                              write_to_xmp, write_text_file, write_to_database,
                              debug_logging):
        """Process single matches with metadata handling"""
        try:
            destination_paths = []
            matched_keywords = []

            for keyword, folder in matches:
                folder_path = os.path.join(output_dir, folder)
                os.makedirs(folder_path, exist_ok=True)
                
                dest_path = os.path.join(folder_path, os.path.basename(input_filepath))

                # Handle file operation
                if not self._perform_file_operation(
                    images, input_filepath, dest_path, 
                    action, is_first=not destination_paths
                ):
                    continue

                # Create metadata for match
                metadata = {
                    'basic': {
                        'keywords': list(keywords)
                    },
                    'sort_info': {
                        'sort_method': 'single_match',
                        'matched_keywords': [keyword],
                        'sort_date': datetime.now().isoformat()
                    }
                }

                # Set targets based on user preferences
                targets = []
                if write_to_xmp: targets.append('xmp')
                if embed_metadata: targets.append('embedded')
                if write_text_file: targets.append('txt')
                if write_to_database: targets.append('db')
                
                if targets:
                    # Set resource identifier (important for proper XMP handling)
                    filename = os.path.basename(dest_path)
                    resource_uri = f"file:///{filename}"
                    self.metadata_service.set_resource_identifier(resource_uri)
                    
                    # Write metadata
                    write_results = self.metadata_service.write_metadata(dest_path, metadata, targets=targets)
                    
                    # Log results
                    if debug_logging:
                        success_targets = [t for t, success in write_results.items() if success]
                        if success_targets:
                            print(f"[KeywordSorter] Successfully wrote metadata to: {', '.join(success_targets)}")
                        else:
                            print("[KeywordSorter] Failed to write metadata to any target")

                destination_paths.append(dest_path)
                matched_keywords.append(keyword)

            return (
                images,
                ", ".join(destination_paths),
                ", ".join(matched_keywords)
            )

        except Exception as e:
            print(f"[KeywordSorter] Single match processing error: {str(e)}")
            return (images, "", "")

    def _perform_file_operation(self, images, src_path, dest_path, 
                              action, is_first=True):
        """Perform file operation with proper error handling"""
        try:
            if action == "move" and is_first:
                shutil.move(src_path, dest_path)
                # Move sidecar files
                for ext in ['.txt', '.json', '.xmp']:
                    sidecar = os.path.splitext(src_path)[0] + ext
                    if os.path.exists(sidecar):
                        shutil.move(sidecar, os.path.splitext(dest_path)[0] + ext)
            elif action == "copy":
                shutil.copy2(src_path, dest_path)
                # Copy sidecar files
                for ext in ['.txt', '.json', '.xmp']:
                    sidecar = os.path.splitext(src_path)[0] + ext
                    if os.path.exists(sidecar):
                        shutil.copy2(sidecar, os.path.splitext(dest_path)[0] + ext)
            else:  # save_new
                img_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(dest_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            return True

        except Exception as e:
            print(f"[KeywordSorter] File operation error: {str(e)}")
            return False

    def cleanup(self):
        """Clean up through metadata service"""
        if hasattr(self, 'metadata_service'):
            self.metadata_service.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()


NODE_CLASS_MAPPINGS = {
    "Eric_Keyword_Sorter_V6": Keyword_Sorter_Node_V6
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eric_Keyword_Sorter_V6": "Eric's Multi-Keyword Sorter V6"
}