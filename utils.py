import os


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def import_class(class_path):
    """
    Imports a class having dynamic name/path.

    Args:
        class_path: str
            Path to class (exp: "model.MmdetDetectionModel")
    Returns:
        mod: class with given path
    """
    components = class_path.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
