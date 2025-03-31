# ComfyUI SISO LoRA Generator Node

## Overview

This custom node provides a wrapper within ComfyUI to execute the `siso_generation_sdxl.py` script (from the original SISO project). It allows you to generate subject-specific LoRA files based on the SISO (Subject-driven Image Synthesis and Optimization) method directly from the ComfyUI interface by providing the necessary parameters.

**Important:** This node executes the external Python script as a blocking subprocess. This means the **ComfyUI interface will freeze** while the script is running. The duration depends on the number of epochs and your hardware.

## Prerequisites

1.  **ComfyUI:** A working installation of ComfyUI.
2.  **Python Environment:** A Python environment (e.g., venv, conda) accessible by ComfyUI where the dependencies can be installed.
3.  **Git:** Required for cloning this repository.
4.  **SISO Project Files:** You need the `siso_generation_sdxl.py` script and its associated utility files (like `utils/dino_utils.py`, `utils/ir_features_utils.py`) from the original SISO project accessible on your system.
5.  **IR Features Weights:** **CRITICAL:** The SISO method requires specific pre-trained weights for IR feature extraction. You **must manually download** the `ir_features.pth` file (refer to the original SISO project repository for download instructions/links) and place it somewhere accessible on your system. Note the full path to this file.
6.  **GPU VRAM:** **WARNING:** The underlying SISO script performs an iterative optimization process that is computationally intensive and requires significant GPU VRAM, similar to fine-tuning. **16GB+ VRAM is likely recommended**, potentially more depending on resolution and rank. Insufficient VRAM will likely lead to "CUDA out of memory" errors.
7.  **Version Compatibility:** This node and the underlying script may have specific version requirements for libraries like `torch`, `diffusers`, `transformers`, etc. Check the `requirements.txt` file and be prepared to potentially adjust versions in your environment if you encounter compatibility issues.
8.  **Device Targeting (CUDA/MPS/XPU):** This node launches the external `siso_generation_sdxl.py` script. Device selection (GPU vs CPU, specific backend like CUDA, MPS, or XPU) is handled *within that script*, typically via the `accelerate` library or `torch.device` settings based on your environment's configuration (e.g., `accelerate config`). Ensure your environment is correctly configured for your desired hardware accelerator *before* running ComfyUI and this node. The node itself does not include device selection logic.

## Installation

1.  Open a terminal or command prompt.
2.  Navigate to your ComfyUI installation directory, then into the `custom_nodes` sub-directory:
    ```bash
    cd /path/to/ComfyUI/custom_nodes/
    ```
3.  Clone this repository:
    ```bash
    git clone <repository_url> ComfyUI-SISO-Generator
    ```
    (Replace `<repository_url>` with the actual URL of this node's repository)
4.  Navigate into the newly cloned directory:
    ```bash
    cd ComfyUI-SISO-Generator
    ```
5.  **Install Dependencies into the Correct Environment:** This is crucial. You need to install the Python packages listed in `requirements.txt` into the *specific Python environment that your ComfyUI installation uses*. How you do this depends on your ComfyUI setup:
    *   **If using a virtual environment (venv/conda):** Activate the environment first, then run pip install.
        ```bash
        # Example for venv
        source /path/to/your/comfyui_env/bin/activate
        # Example for conda
        # conda activate your_comfyui_env_name
        
        pip install -r requirements.txt
        ```
    *   **If using ComfyUI Portable/Standalone (with embedded Python):** You might need to run the `python_embeded\python.exe` or similar executable directly with the pip command. Check your ComfyUI documentation. It might look something like:
        ```bash
        # Example (Windows - path may vary)
        C:\path\to\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
        # Example (Linux - path may vary)
        # /path/to/ComfyUI_linux/python_embeded/bin/python -m pip install -r requirements.txt
        ```
    *   **If using ComfyUI Manager:** While ComfyUI Manager can install nodes, it might not handle complex dependencies like these perfectly. Installing manually via `pip` as described above is often more reliable for nodes with extensive requirements.
    *   **Verify Installation:** After running pip, ensure there were no major errors.
6.  Restart ComfyUI fully after installing the dependencies.

## Usage

1.  Launch ComfyUI.
2.  Add the **"SISO LoRA Generator (Blocking)"** node (found under the "SISO" category) to your workflow.
3.  Configure the node's inputs:
    *   **`subject_image`**: Connect the output of a `Load Image` node containing your subject image.
    *   **`prompt`**: Enter the text prompt you want to associate with the subject for LoRA generation (e.g., "photo of sks subject").
    *   **`weights_output_dir`**: Specify the **absolute directory path** where the generated LoRA file (`pytorch_lora_weights.safetensors`) should be saved. Relative paths might be unreliable depending on where ComfyUI is launched from. The node will attempt to create this directory if it doesn't exist. Ensure write permissions.
    *   **`base_model_path`**: Provide the **absolute path** to your base SDXL model checkpoint file or its Hugging Face repository ID (e.g., `stabilityai/stable-diffusion-xl-base-1.0`). The underlying script needs to access this.
    *   **`siso_script_path`**: Provide the **absolute path** to the `siso_generation_sdxl.py` script file on your system.
    *   **`ir_features_path`**: **CRUCIAL:** Provide the **absolute path** to the `ir_features.pth` file you downloaded manually.
    *   **Other Parameters**: Adjust the script parameters like `num_train_epochs`, `learning_rate`, `rank`, feature weights, early stopping settings, `resolution`, and `seed` as needed.
4.  Connect a trigger to the **`execute`** input. A simple way is to add a `Primitive` node, set its value to `True`, and connect it.
5.  Queue the prompt (Ctrl+Enter or click "Queue Prompt").
6.  **Wait:** ComfyUI's interface will now **freeze**. Monitor the terminal/console where you launched ComfyUI for log messages from the node (prefixed with `[SISO Node]`) and potentially from the script itself. This process can take several minutes or longer.
7.  **Check Results:** Once the process completes (ComfyUI unfreezes), check the `status` output of the node:
    *   **"Success: LoRA saved to ..."**: The process completed successfully. The `lora_path` output contains the full path to the generated `.safetensors` file.
    *   **"Failed: ..."**: An error occurred. Check the status message and the console output for details.
8.  **Use the LoRA:** If successful, you can now use the `lora_path` output in a separate inference workflow (or queue it later in the same workflow). Connect the `lora_path` string output to the `lora_name` input of a standard `Load LoRA` node.

## Troubleshooting

*   **Errors during execution:** Check the console/terminal where ComfyUI is running. The node attempts to print detailed error messages from the script's `stderr`.
*   **`FileNotFoundError`:** Double-check the paths provided for `siso_script_path`, `ir_features_path`, and `base_model_path`. Ensure the files exist and ComfyUI has permission to read them.
*   **`CUDA out of memory`:** The most common issue. Your GPU doesn't have enough VRAM. Try reducing the `resolution` or `rank` parameters in the node. Close other GPU-intensive applications.
*   **Dependency/Import Errors:** Ensure you installed the requirements correctly into ComfyUI's Python environment. Check for version conflicts.
*   **Script finished but LoRA not found:** Verify the `weights_output_dir` path is correct and that the script has permission to write there. Check the script's console output for any specific saving errors.

## Disclaimer

This node acts as a wrapper to execute an external script. Its successful operation depends heavily on the correct setup of the user's environment, the validity of the provided paths, the availability of sufficient hardware resources (especially GPU VRAM), and the correct functioning of the underlying `siso_generation_sdxl.py` script and its dependencies.