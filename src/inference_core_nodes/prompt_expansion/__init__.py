from .prompt_expansion import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    MODEL_FOLDER_NAME,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _add_folder_paths():
    from pathlib import Path
    import folder_paths

    folder_paths.add_model_folder_path(
        MODEL_FOLDER_NAME,
        str(Path(folder_paths.models_dir).joinpath(MODEL_FOLDER_NAME)),
    )


_add_folder_paths()
