# Plan: Blocking Script Runner Node for SISO LoRA Generation (v2 - Incorporating Feedback)

## 1. Goal

Create a ComfyUI custom node (`SISO LoRA Generator`) that allows users to configure parameters and execute the `siso_generation_sdxl.py` script from within the ComfyUI interface. This node will generate a subject-specific LoRA file. The ComfyUI interface will freeze during script execution. The node package will include mechanisms for installing necessary dependencies and provide guidance on usage.

## 2. Dependency Identification & Version Compatibility

*   Analyze `siso_generation_sdxl.py`, `utils/dino_utils.py`, `utils/ir_features_utils.py`, and the project's `environment.yml` to list all required Python packages (e.g., `torch`, `diffusers`, `transformers`, `peft`, `timm`, `open_clip`, `kornia`, `accelerate`, `numpy`, `datasets`, `Pillow`, etc.).
*   **Crucially, identify and document specific version requirements**, especially for `torch`, `diffusers`, and `transformers`, as SISO might rely on recent features. These versions should be specified in `requirements.txt`.
*   The `README.md` will highlight any critical version compatibility notes.

## 3. Node Package Structure

*   Create a directory for the custom node (e.g., `ComfyUI-SISO-Generator`).
*   Place the node implementation file (`SISO_LoRA_Generator.py`) inside.
*   Create a `requirements.txt` file listing all identified dependencies with specific versions where necessary.
*   *(Optional)* Include a simple `install.sh` (for Linux/macOS) and/or `install.bat` (for Windows) script that runs `pip install -r requirements.txt`.
*   Include a `README.md` with clear installation instructions, usage guidance, **GPU VRAM requirement warnings**, notes on **version compatibility**, and instructions for obtaining the **IR features weights**.

## 4. Node Implementation (`SISO_LoRA_Generator.py`)

*   **Node Class:** Define a Python class (e.g., `SISOLoRAGenerator`).
*   **Inputs:** Define inputs corresponding to the script's key arguments:
    *   `subject_image` (`IMAGE`): The input subject image.
    *   `prompt` (`STRING`): The prompt for LoRA generation.
    *   `weights_output_dir` (`STRING`): Directory where the script will save the LoRA file. **This is the `--weights_output_dir` argument for the script.**
    *   `base_model_path` (`STRING`): Path to the base SDXL model checkpoint (`--pretrained_model_name_or_path`).
    *   `siso_script_path` (`STRING`): Path to the `siso_generation_sdxl.py` script.
    *   `ir_features_path` (`STRING`): **Critical:** Path to the pre-downloaded IR features weights file (`--ir_features_path`). Emphasize in UI/docs that this needs manual download.
    *   Script parameters exposed as widgets with appropriate defaults:
        *   `num_train_epochs` (INT)
        *   `learning_rate` (FLOAT)
        *   `dino_features_weight` (FLOAT)
        *   `ir_features_weight` (FLOAT)
        *   `seed` (INT)
        *   `resolution` (INT)
        *   `rank` (INT)
        *   `early_stopping_max_count` (INT) (`--early_stopping_max_count`)
        *   `early_stopping_threshold_percentage` (INT) (`--early_stopping_threshold_percentage`)
        *   *(Add other relevant script args as needed)*
    *   `execute` (`BOOLEAN`, `forceInput=True`): A trigger to start the script execution.
*   **Outputs:**
    *   `lora_path` (`STRING`): The full path to the generated `pytorch_lora_weights.safetensors` file upon successful completion.
    *   `status` (`STRING`): A message indicating "Success" or "Failed: [Error details]".
*   **Execution Logic:**
    *   When triggered, the node will:
        *   Verify the `siso_script_path` and `ir_features_path` point to valid files.
        *   Save the input `subject_image` tensor to a temporary file (e.g., using `PIL`).
        *   Construct the command-line arguments needed to run `siso_generation_sdxl.py`, using the node's input values and the temporary image path. Ensure `--save_weights` is included and `--weights_output_dir` is set correctly.
        *   Use Python's `subprocess.run()` to execute the script (e.g., `python [siso_script_path] --subject_image_path [temp_image_path] --weights_output_dir [weights_output_dir] --save_weights ...`), capturing `stdout` and `stderr`, and setting `check=False` (to handle errors manually). This call will block ComfyUI.
        *   **Error Handling:**
            *   Check the script's `returncode`. If non-zero, parse `stderr` for common errors (e.g., "CUDA out of memory", file not found, import errors) and return a specific "Failed: [Parsed Error]" status. If parsing fails, return the raw `stderr`.
            *   If `subprocess.run()` itself raises an exception (e.g., script not executable), catch it and return "Failed: [Exception details]".
        *   **Success Path:**
            *   If the return code is 0, determine the expected output LoRA file path: `os.path.join(weights_output_dir, 'pytorch_lora_weights.safetensors')`.
            *   Verify this LoRA file exists using `os.path.exists()`. If not, return "Failed: Script finished but LoRA file not found at expected location."
            *   If the file exists, return the path in `lora_path` and "Success" in `status`.
        *   Clean up the temporary image file in a `finally` block.

## 5. Installation & Usage

*   Users clone or download the `ComfyUI-SISO-Generator` directory into their `ComfyUI/custom_nodes/` folder.
*   Users **must manually download the required IR features weights** and note the path.
*   Users run the installation script (`install.sh`/`install.bat`) or manually run `pip install -r requirements.txt` in their ComfyUI environment's terminal. **Pay attention to version compatibility notes.**
*   Restart ComfyUI.
*   Add the `SISO LoRA Generator` node to a workflow.
*   Configure the inputs, **ensuring the `ir_features_path` is correct**.
*   Trigger the node execution. ComfyUI will freeze while the script runs. **Ensure sufficient GPU VRAM is available.**
*   Once finished, check the `status` output. If successful, the `lora_path` output can be used in a separate inference workflow (or queued later) with a standard `Load LoRA` node.

## 6. Workflow Diagram

```mermaid
graph TD
    subgraph "Setup (Manual User Action)"
        direction LR
        DownloadNode[Download/Clone Node Files] --> PlaceInCustomNodes[Place in custom_nodes];
        DownloadIRWeights[Download IR Weights] --> NoteIRPath[Note IR Weights Path];
        PlaceInCustomNodes --> RunInstallScript[Run install.sh/bat or pip install (Check Versions)];
        RunInstallScript --> RestartComfyUI;
    end

    subgraph "ComfyUI Workflow"
        direction LR

        subgraph "LoRA Generation (Blocking)"
            LoadSubjectImage --> SISO_ScriptRunner;
            PromptWidget[Prompt Widget] --> SISO_ScriptRunner;
            WeightsOutputDirWidget[Weights Output Dir Widget] --> SISO_ScriptRunner;
            BaseModelWidget[Base Model Path Widget] --> SISO_ScriptRunner;
            SisoScriptPathWidget[SISO Script Path Widget] --> SISO_ScriptRunner;
            IRFeaturesPathWidget[IR Features Path Widget] --> SISO_ScriptRunner;
            ParamsWidgets[Param Widgets (Epochs, LR, EarlyStop, etc.)] --> SISO_ScriptRunner;
            ExecuteTrigger[Execute Trigger] --> SISO_ScriptRunner;
            SISO_ScriptRunner(SISO LoRA Generator Node) -- LoRA Path --> LoRA_Path_Output[LoRA Path Output];
            SISO_ScriptRunner -- Status --> Status_Output[Status Output];
            style SISO_ScriptRunner fill:#f9d,stroke:#333,stroke-width:2px
        end

        subgraph "Inference (Separate Execution/Graph)"
            LoadCheckpoint --> ApplyLoRA;
            LoadLoRA[Load LoRA (using LoRA_Path_Output)] --> ApplyLoRA;
            ApplyLoRA[Apply LoRA (Model, CLIP)] --> KSampler;
            CLIPTextEncodePositive[CLIP Text Encode (Prompt)] --> KSampler;
            CLIPTextEncodeNegative[CLIP Text Encode (Negative)] --> KSampler;
            EmptyLatent --> KSampler;
            LoadCheckpoint -- VAE --> VAEDecode;
            KSampler -- Latent --> VAEDecode;
            VAEDecode --> SaveImage;
        end

        LoRA_Path_Output --> LoadLoRA;

        classDef comfyNode fill:#eee,stroke:#333,stroke-width:1px;
        class LoadSubjectImage,PromptWidget,WeightsOutputDirWidget,BaseModelWidget,SisoScriptPathWidget,IRFeaturesPathWidget,ParamsWidgets,ExecuteTrigger,LoRA_Path_Output,Status_Output,LoadCheckpoint,LoadLoRA,ApplyLoRA,CLIPTextEncodePositive,CLIPTextEncodeNegative,EmptyLatent,KSampler,VAEDecode,SaveImage comfyNode;

    end

    subgraph "External Process (Launched by Node - Blocking)"
      SISO_ScriptRunner -- Runs & Waits --> PythonSubprocess[`python siso_generation_sdxl.py ... --save_weights --weights_output_dir ...`];
      PythonSubprocess --> GeneratedLoRA["pytorch_lora_weights.safetensors"];
      style PythonSubprocess fill:#ccf,stroke:#333,stroke-width:1px
    end