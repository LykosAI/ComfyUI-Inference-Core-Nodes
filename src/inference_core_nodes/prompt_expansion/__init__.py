from .prompt_expansion import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    MODEL_FOLDER_NAME,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _add_folder_paths():
    from pathlib import Path
    import folder_paths
    
    folder_names_and_paths = folder_paths.folder_names_and_paths
    
    model_folder_path = str(Path(folder_paths.models_dir).joinpath(MODEL_FOLDER_NAME))
    supported_extensions = {'.ckpt', '.pt', '.bin', '.pth', '.safetensors'}
    
    if MODEL_FOLDER_NAME in folder_names_and_paths:
        folder_names_and_paths[MODEL_FOLDER_NAME][0].append(model_folder_path)
    else:
        folder_names_and_paths[MODEL_FOLDER_NAME] = ([model_folder_path], supported_extensions)


_add_folder_paths()
