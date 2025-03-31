import torch
import numpy as np
from PIL import Image
import os
import subprocess
import tempfile
import shlex # For safe command line argument construction
import inspect # To get the node's file path

# Helper function to convert tensor to PIL Image
def tensor_to_pil(tensor):
    image_np = tensor.cpu().numpy().squeeze()
    if image_np.ndim == 3:
        image_np = image_np.transpose(1, 2, 0)  # CHW to HWC
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

class SISOLoRAGenerator:
    """
    ComfyUI node to run the siso_generation_sdxl.py script for LoRA generation.
    Blocks the UI during execution.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "photo of sks subject"}),
                "weights_output_dir": ("STRING", {"default": "output/siso_lora"}), # Recommend absolute path in UI/docs
                "base_model_path": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}), # Recommend absolute path
                # Defaults assume node is symlinked into custom_nodes from root repo containing these files/dirs
                "siso_script_path": ("STRING", {"default": "siso_generation_sdxl.py"}),
                "ir_features_path": ("STRING", {"default": "models/ir_weights/ir_features.pth"}),
                # --- Script Parameters ---
                "num_train_epochs": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-7, "max": 1e-2, "step": 1e-6}),
                "rank": ("INT", {"default": 4, "min": 1, "max": 128}),
                "dino_features_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "ir_features_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "early_stopping_max_count": ("INT", {"default": 5, "min": 1, "max": 100}),
                "early_stopping_threshold_percentage": ("INT", {"default": 3, "min": 0, "max": 100}),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 50}), # Corresponds to optimization steps in script
                "resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "execute": ("BOOLEAN", {"default": False, "forceInput": True}), # Trigger
            },
            "optional": {
                 # Add other optional script args here if needed, e.g.:
                 # "adam_weight_decay": ("FLOAT", {"default": 1e-2}),
                 # "train_text_encoder": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "status")
    FUNCTION = "generate_lora"
    CATEGORY = "SISO"

    def generate_lora(self, subject_image, prompt, weights_output_dir, base_model_path,
                      siso_script_path, ir_features_path, num_train_epochs, learning_rate,
                      rank, dino_features_weight, ir_features_weight, early_stopping_max_count,
                      early_stopping_threshold_percentage, num_inference_steps, resolution, seed, execute, **kwargs):

        if not execute:
            # Return None for path and a non-error status when not executing
            return (None, "Idle: Set execute to True")

        status = "Starting..."
        lora_output_path = None
        temp_image_path = None
        
        # Get the directory containing this node file
        node_file_path = inspect.getfile(self.__class__)
        node_dir = os.path.dirname(node_file_path)
        # Assume the main project root is one level up from the node dir (e.g., ../ from comfyui_node/)
        project_root = os.path.abspath(os.path.join(node_dir, '..'))

        try:
            # Resolve potentially relative paths to absolute paths based on project root
            abs_siso_script_path = siso_script_path if os.path.isabs(siso_script_path) else os.path.join(project_root, siso_script_path)
            abs_ir_features_path = ir_features_path if os.path.isabs(ir_features_path) else os.path.join(project_root, ir_features_path)
            abs_weights_output_dir = weights_output_dir if os.path.isabs(weights_output_dir) else os.path.join(project_root, weights_output_dir)
            # base_model_path can be a HF ID or a path, handle path case
            abs_base_model_path = base_model_path
            if not os.path.exists(base_model_path) and os.path.isfile(os.path.join(project_root, base_model_path)):
                 # If it's not an existing path/ID, but exists relative to project root, make it absolute
                 abs_base_model_path = os.path.join(project_root, base_model_path)
            elif os.path.exists(base_model_path) and not os.path.isabs(base_model_path):
                 # If it exists but is relative, make it absolute based on CWD (standard behavior)
                 abs_base_model_path = os.path.abspath(base_model_path)


            # 1. Verify resolved script and IR paths
            if not os.path.exists(abs_siso_script_path):
                raise FileNotFoundError(f"SISO script not found at resolved path: {abs_siso_script_path} (based on node location)")
            if not os.path.exists(abs_ir_features_path):
                raise FileNotFoundError(f"IR features file not found at resolved path: {abs_ir_features_path} (based on node location)")

            # 2. Save input image to temporary file
            img_pil = tensor_to_pil(subject_image)
            # Use a temporary directory to handle potential permission issues
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_image_path = os.path.join(tmpdir, "siso_subject_image.png")
                img_pil.save(temp_image_path)
                status = f"Saved temp image to {temp_image_path}"
                print(f"[SISO Node] {status}") # Log progress

                # 3. Construct command line arguments using absolute paths
                # Ensure output directory exists
                os.makedirs(abs_weights_output_dir, exist_ok=True)

                cmd = [
                    "python", # Assumes python is in PATH and points to the correct env
                    abs_siso_script_path, # Use absolute path
                    "--pretrained_model_name_or_path", abs_base_model_path, # Use resolved path/ID
                    "--subject_image_path", temp_image_path, # Temp path is already absolute
                    "--prompt", prompt,
                    "--weights_output_dir", abs_weights_output_dir, # Use absolute path
                    "--ir_features_path", abs_ir_features_path, # Use absolute path
                    "--num_train_epochs", str(num_train_epochs),
                    "--learning_rate", str(learning_rate),
                    "--rank", str(rank),
                    "--dino_features_weight", str(dino_features_weight),
                    "--ir_features_weight", str(ir_features_weight),
                    "--early_stopping_max_count", str(early_stopping_max_count),
                    "--early_stopping_threshold_percentage", str(early_stopping_threshold_percentage),
                    "--num_inference_steps", str(num_inference_steps),
                    "--resolution", str(resolution),
                    "--seed", str(seed),
                    "--save_weights", # Always save weights
                    # Add other optional args from kwargs if they exist
                ]
                # Example for optional args:
                # if 'adam_weight_decay' in kwargs:
                #    cmd.extend(["--adam_weight_decay", str(kwargs['adam_weight_decay'])])
                # if kwargs.get('train_text_encoder', False):
                #    cmd.append("--train_text_encoder")

                status = f"Running command: {' '.join(shlex.quote(c) for c in cmd)}"
                print(f"[SISO Node] {status}")

                # 4. Run the subprocess (blocking)
                # Using check=False and capturing output to handle errors manually
                process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8') # Specify encoding

                # 5. Error Handling
                if process.returncode != 0:
                    error_message = f"Script failed with code {process.returncode}.\n"
                    # Decode stderr if it's bytes, otherwise use as is
                    stderr_str = process.stderr if isinstance(process.stderr, str) else process.stderr.decode('utf-8', errors='ignore')
                    stdout_str = process.stdout if isinstance(process.stdout, str) else process.stdout.decode('utf-8', errors='ignore')

                    error_message += f"Stderr:\n{stderr_str}\n"
                    error_message += f"Stdout:\n{stdout_str}"

                    # Basic parsing for common errors
                    if "CUDA out of memory" in stderr_str:
                        status = "Failed: CUDA out of memory. Try reducing resolution or rank."
                    elif "FileNotFoundError" in stderr_str:
                         status = f"Failed: FileNotFoundError in script. Check paths. Stderr: {stderr_str[:500]}" # Truncate long errors
                    else:
                         status = f"Failed: Script execution error. Check console/stderr. Stderr: {stderr_str[:500]}"
                    print(f"[SISO Node] Error: {error_message}") # Log full error
                    # Don't raise exception here, return status string instead
                    return (None, status)


                # 6. Success Path
                status = "Script finished successfully."
                print(f"[SISO Node] {status}")
                # Check for output file using the absolute output directory path
                expected_lora_file = os.path.join(abs_weights_output_dir, 'pytorch_lora_weights.safetensors')

                if not os.path.exists(expected_lora_file):
                    status = f"Failed: Script finished but LoRA file not found at {expected_lora_file}"
                    print(f"[SISO Node] Error: {status}")
                    return (None, status) # Return None path and error status

                lora_output_path = expected_lora_file
                status = f"Success: LoRA saved to {lora_output_path}"
                print(f"[SISO Node] {status}")

        except Exception as e:
            status = f"Failed: Node error - {type(e).__name__}: {str(e)}"
            print(f"[SISO Node] Exception: {status}")
            lora_output_path = None # Ensure path is None on failure
            # Return None path and error status
            return (None, status)

        # If we reach here, it means success (or handled failure within try)
        return (lora_output_path, status)


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SISOLoRAGenerator": SISOLoRAGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SISOLoRAGenerator": "SISO LoRA Generator (Blocking)"
}