__all__ = ("__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")

__version__ = "0.3.0"


def _get_node_mappings():
    import importlib
    import pkgutil
    import re

    def _transform_node_name(node_name: str) -> str:
        """
        Adds our global prefix 'Inference_Core_' to a node name to prevent collisions.
        Also replaces spaces with underscores.
        """
        # Remove multiple spaces
        node_name = re.sub(" +", " ", node_name)
        # Replace spaces with underscores
        node_name = node_name.replace(" ", "_")
        # Add prefix
        node_name = f"Inference_Core_{node_name}"

        return node_name

    def _transform_node_display_name(display_name: str) -> str:
        """
        Adds our global prefix '[Inference.Core]' to a node name for identification.
        """
        return f"[Inference.Core] {display_name}"

    node_class_mappings = {}
    node_display_name_mappings = {}

    for submodule_info in pkgutil.iter_modules(__path__):
        # import
        submodule = importlib.import_module(f"{__name__}.{submodule_info.name}")

        node_class_mappings.update(getattr(submodule, "NODE_CLASS_MAPPINGS"))
        node_display_name_mappings.update(
            getattr(submodule, "NODE_DISPLAY_NAME_MAPPINGS")
        )

    # Transform the node names and display names
    node_class_mappings = {
        _transform_node_name(node_name): node_class
        for node_name, node_class in node_class_mappings.items()
    }

    node_display_name_mappings = {
        _transform_node_name(node_name): _transform_node_display_name(display_name)
        for node_name, display_name in node_display_name_mappings.items()
    }

    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _get_node_mappings()
