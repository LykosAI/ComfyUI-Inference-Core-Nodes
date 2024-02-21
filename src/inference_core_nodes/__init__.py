__all__ = ("__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")

__version__ = "0.1.0"


def _get_node_mappings():
    import importlib
    import pkgutil

    node_class_mappings = {}
    node_display_name_mappings = {}

    for submodule_info in pkgutil.iter_modules(__path__):
        # import
        submodule = importlib.import_module(f"{__name__}.{submodule_info.name}")

        node_class_mappings.update(getattr(submodule, "NODE_CLASS_MAPPINGS"))
        node_display_name_mappings.update(
            getattr(submodule, "NODE_DISPLAY_NAME_MAPPINGS")
        )

    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _get_node_mappings()
