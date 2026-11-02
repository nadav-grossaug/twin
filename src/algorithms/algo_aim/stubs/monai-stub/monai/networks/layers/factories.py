from monai import _Unavailable


class _Factory:
    def __getattr__(self, name):
        return _Unavailable

    def __getitem__(self, name):
        return _Unavailable


Act = _Factory()
