# SSM Analyzer - Run Guide

## 1. Introduction

This document provides instructions for setting up and running the Surrogate Safety Measures (SSM) Analyzer script (`ssm_analyzer.py`). The script is designed to process drone video footage to detect vehicles, track their movements, and calculate various safety metrics such as Time-to-Collision (TTC), Post-Encroachment Time (PET), and Gap Time. The results are outputted to a CSV file, and an annotated video can also be generated.

## 2. Prerequisites

Before you begin, ensure your system meets the following requirements:

-   **Python**: Version 3.8 or higher.
-   **NVIDIA GPU**: An NVIDIA GPU with CUDA support is highly recommended for performance, especially for the object detection (YOLO) and tracking (DeepSORT) components.
-   **NVIDIA CUDA Toolkit & cuDNN**: If using a GPU, you must have the appropriate versions of the NVIDIA CUDA Toolkit and cuDNN installed. These versions need to be compatible with the PyTorch version you will install.
-   **Git**: Required for cloning repositories (e.g., this project if applicable, and potentially DeepSORT).

## 3. Setup Instructions

Follow these steps to set up the necessary environment and dependencies.

### 3.1. Clone the Repository / Download Files

If you have a Git repository URL for this project:
```bash
git clone <repository_url>
cd <repository_directory>
```
Alternatively, download and extract the script files (`ssm_analyzer.py`, `requirements.txt`, etc.) into a project directory and navigate into it.

### 3.2. Create a Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**Using `venv` (standard Python):**
```bash
# Navigate to your project directory first
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# For Windows (Command Prompt): .venv\Scripts\activate
# For Windows (PowerShell):   .venv\Scripts\Activate.ps1
```

**Or using `conda` (Anaconda/Miniconda):**
```bash
conda create -n ssm_env python=3.9  # You can choose a specific Python version
conda activate ssm_env
```

### 3.3. Install PyTorch with CUDA Support

For optimal performance, install PyTorch with CUDA support that matches your system's NVIDIA driver and CUDA Toolkit version.

1.  Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Use the configuration tool on the website to find the precise installation command for your system (OS, package manager, CUDA version).

**Example command (verify on the PyTorch website for your specific setup!):**
This example is for a system with CUDA 11.8. Your command might differ.
```bash
# Example for CUDA 11.8 (Verify on PyTorch website for your system!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you do not have an NVIDIA GPU or do not wish to install CUDA, you can install the CPU-only version of PyTorch (the PyTorch website will also provide this command).

### 3.4. Install Other Python Dependencies

Once PyTorch is installed and your virtual environment is active, install the remaining dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3.5. Install DeepSORT

DeepSORT is not available as a standard pip package and often requires installation from a specific source repository.

-   You will need a PyTorch-compatible DeepSORT implementation. A common approach is to use a well-maintained fork of `deep_sort_pytorch`.
-   **Example (you might need to find an up-to-date and compatible fork):**
    ```bash
    # Example: Clone a popular repository (check for alternatives suitable for recent YOLO versions)
    # git clone https://github.com/ZQPei/deep_sort_pytorch.git
    # cd deep_sort_pytorch
    # If the chosen DeepSORT repository has its own requirements.txt, install them:
    # pip install -r requirements.txt 
    ```
-   **Accessibility**: Ensure the DeepSORT library is accessible in your Python path.
    -   If the library includes a `setup.py` file, you might be able to install it into your virtual environment using `python setup.py install` or `pip install .` from within its directory.
    -   Alternatively, you may need to add the directory containing the `deep_sort_pytorch` module (or your chosen library's equivalent) to your `PYTHONPATH` environment variable.
-   **Import Path**: The `ssm_analyzer.py` script attempts to import DeepSORT components using `from deep_sort_pytorch.utils.parser import get_config` and `from deep_sort_pytorch.deep_sort import DeepSort`. If your chosen DeepSORT library has a different module structure, you may need to adjust these import statements in `ssm_analyzer.py`.

### 3.6. Obtain Model Weights and Configuration

**YOLO Model:**
-   The YOLO model specified in `ssm_analyzer.py` (e.g., `yolov8x.pt`) will typically be downloaded automatically by the `ultralytics` library the first time it's needed if it's not found in a local cache or standard model directory. No manual download is usually required for standard YOLOv8 models.

**DeepSORT Re-identification (ReID) Model:**
-   The DeepSORT algorithm relies on a Re-identification (ReID) model to help maintain object identities. A common model is `osnet_x0_25_msmt17.pt`.
-   **Download**: You need to download this ReID model checkpoint file. You can search online for "osnet_x0_25_msmt17.pt download". A common source is often linked in DeepSORT repository documentation.
    *(Note: As a language model, I cannot provide direct download links, but a web search should yield results from academic or project sites.)*
-   **Placement**: Place the downloaded ReID model file (e.g., `osnet_x0_25_msmt17.pt`) in a location accessible by the script.
-   **Path Configuration**: The path to this checkpoint is specified in `ssm_analyzer.py` within the `initialize_deepsort_tracker` function (argument `model_path`). The default is often `"osnet_x0_25_msmt17.pt"`, implying it should be in the same directory as the script or a path recognizable by the DeepSORT library. If you place it elsewhere, you **must** update this path in the script.

**DeepSORT Configuration File (Optional but Recommended):**
-   Some DeepSORT implementations use a YAML configuration file (e.g., `deep_sort.yaml` or often found at `deep_sort_pytorch/configs/deep_sort.yaml` within the cloned DeepSORT repository).
-   The `ssm_analyzer.py` script (specifically in `initialize_deepsort_tracker`) might attempt to load settings from such a file using `get_config()`.
-   If your chosen DeepSORT library relies on this file, ensure it is present and correctly configured. The script attempts to use sensible defaults if the config loading fails or specific attributes are missing, but behavior might be suboptimal.

## 4. Running the Script

Once the setup is complete, you can run the script from your terminal or command prompt. Ensure your virtual environment is activated.

### Command-Line Interface (CLI) Usage:

```bash
python ssm_analyzer.py --input_video /path/to/your_video.MP4 \
                       --output_csv /path/to/your_results.csv \
                       --output_video /path/to/your_annotated_video.mp4 \
                       --pixels_per_meter YOUR_CALIBRATED_VALUE
```

### Arguments Explanation:

-   `--input_video INPUT_VIDEO`: **(Required)** Path to the input drone video file (e.g., `videos/video_0058.MP4`).
-   `--output_csv OUTPUT_CSV`: **(Required)** Path to save the CSV file that will contain the trajectory data and calculated SSM results.
-   `--output_video OUTPUT_VIDEO`: **(Optional)** Path to save the annotated output video. If this argument is not provided, no annotated video will be generated.
-   `--pixels_per_meter PIXELS_PER_METER`: **(Optional)** The calibration value representing pixels per meter at the drone's altitude in the video. This defaults to the value set in the script (e.g., `10.0`).
    **This value is CRUCIAL for accurate real-world measurements (world coordinates, speeds, and therefore SSMs). You must calibrate this for your specific video setup.**

### CUDA Usage:

-   The script is designed to automatically attempt to use an available NVIDIA GPU if PyTorch (with CUDA support) and compatible NVIDIA drivers/toolkits are correctly installed.
-   During startup, the script will print informational messages indicating:
    -   Whether CUDA is available via PyTorch.
    -   The PyTorch and CUDA versions detected.
    -   Whether the YOLO model is being run on CUDA or CPU.
    -   Whether the DeepSORT tracker will attempt to use CUDA or run on CPU.
-   No specific command-line argument is needed to enable CUDA; it's auto-detected.

## 5. Troubleshooting & Notes

-   **CUDA/Driver Compatibility**: Ensure your NVIDIA drivers, CUDA Toolkit version, and cuDNN version are all compatible with the version of PyTorch you installed. Version mismatches are a common source of issues. Refer to the PyTorch installation guide for compatibility information.
-   **`PYTHONPATH` for DeepSORT**: If you encounter `ModuleNotFoundError` related to DeepSORT, it likely means the DeepSORT library is not in your Python interpreter's search path.
    -   If installed via `setup.py` in your virtual environment, this should generally not be an issue.
    -   If you cloned it into a subfolder, you might need to add the parent directory of the `deep_sort_pytorch` (or equivalent) module to your `PYTHONPATH` environment variable, or restructure your project so it's directly importable.
-   **ReID Model Path**: Double-check the `model_path` for the ReID checkpoint in the `initialize_deepsort_tracker` function within `ssm_analyzer.py`. This must point to the actual location of your downloaded ReID model file (e.g., `osnet_x0_25_msmt17.pt`).
-   **Memory Usage**: Processing high-resolution video, especially with larger YOLO models (like `yolov8x.pt`), can be memory-intensive (both system RAM and GPU VRAM). If you encounter out-of-memory errors:
    -   Consider using a smaller YOLO model (e.g., `yolov8n.pt`, `yolov8s.pt` by changing the model name in `ssm_analyzer.py`).
    -   Process video at a reduced resolution (this script does not currently implement on-the-fly resizing for processing, but it could be added).
    -   Ensure no other memory-heavy applications are running.
-   **DeepSORT Library Variations**: DeepSORT has many forks and variations. The provided script and setup instructions assume a structure similar to common `deep_sort_pytorch` repositories. If you use a significantly different version, you might need to adapt import paths or configuration details in `ssm_analyzer.py`.
-   **Empty Output Video**: If an `--output_video` path is provided but the resulting video file is empty or very small, it might indicate an issue with the `cv2.VideoWriter` initialization (e.g., codec incompatibility, incorrect frame dimensions, or errors during frame processing/writing). Check console messages for errors.

This guide should help you get the SSM Analyzer script up and running. For further issues, consult the specific error messages and documentation for the libraries involved.
