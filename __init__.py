# Metadata_system/__init__.py
import importlib.util
import os
import sys

# Export core functionality for easy access
from .eric_metadata.service import MetadataService
from .eric_metadata.handlers.base import BaseHandler
from .eric_metadata.handlers.xmp import XMPSidecarHandler
from .eric_metadata.handlers.embedded import EmbeddedMetadataHandler
from .eric_metadata.handlers.txt import TxtFileHandler
from .eric_metadata.handlers.db import DatabaseHandler

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Get the nodes directory
def get_nodes_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Import nodes from the nodes directory
nodes_dir = get_nodes_dir("nodes")
for file in os.listdir(nodes_dir):
    if not file.endswith(".py") or file.startswith("__"):
        continue
        
    name = os.path.splitext(file)[0]
    try:
        # Import the module
        imported_module = importlib.import_module(".nodes.{}".format(name), __name__)
        
        # Extract and update node mappings
        if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
            
        if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
            
        print(f"Loaded node module: {name}")
    except Exception as e:
        print(f"Error loading node module {name}: {str(e)}")

# Version info
__version__ = "0.1.0"

# Add web directory for UI components if needed
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]