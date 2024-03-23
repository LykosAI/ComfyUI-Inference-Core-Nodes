# ComfyUI-Inference-Core-Nodes
 
## Installation
1. [Stability Matrix](https://github.com/LykosAI/StabilityMatrix) Extensions Manager
2. [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Manual Installation
1. Clone this repository to `ComfyUI/custom_nodes/`

2. Either:
- Run `install.py` using the venv or preferred python environment.

Or 

(Installs required dependencies and appropriate onnxruntime acceleration via compiled wheels)
- (CUDA 11 or latest stable) Run `pip install -e .[cuda]`
- (CUDA 12) Run `pip install -e .[cuda12]`
- (RoCM) Run `pip install -e .[rocm]`
- (DirectML) Run `pip install -e .[directml]`
- (CPU Only) Run `pip install -e .[cpu]`

Or

(Installs only required dependencies without onnxruntime acceleration)
- Run `pip install -e .`
