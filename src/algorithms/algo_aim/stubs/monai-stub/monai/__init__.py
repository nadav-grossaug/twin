"""Stub for monai. Real monai is unavailable (proxy-blocked); synthcity only needs it for image plugins."""


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise ImportError("monai is stubbed in this project; synthcity image plugins are unsupported")

    def __class_getitem__(cls, item):
        return cls
