def is_field(obj, name):
    """Return whether obj is a dictionary containing name."""
    return isinstance(obj, dict) and name in obj
