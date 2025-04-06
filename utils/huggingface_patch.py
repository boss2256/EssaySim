# utils/huggingface_patch.py
"""
Patch for huggingface_hub to add back the cached_download function
that was removed in newer versions but is still used by sentence-transformers.
"""
import sys
import os
import huggingface_hub

# Check if cached_download is already present (might be in some versions)
if not hasattr(huggingface_hub, 'cached_download'):
    # Add the cached_download function to huggingface_hub module
    def cached_download(*args, **kwargs):
        """
        Compatibility function that redirects to hf_hub_download.
        """
        # Map old arguments to new ones if needed
        if 'cache_dir' not in kwargs and len(args) > 1:
            kwargs['cache_dir'] = args[1]

        # Use current API's hf_hub_download
        return huggingface_hub.hf_hub_download(*args, **kwargs)


    # Add this function to the huggingface_hub module
    huggingface_hub.cached_download = cached_download

    # Also make it available at module level for direct imports
    sys.modules['huggingface_hub'].cached_download = cached_download

    print("Successfully patched huggingface_hub with cached_download function")