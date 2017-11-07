
class objectview(object):
    """Convert dict(or parameters of dict) to object view

    See also:
        - https://goodcode.io/articles/python-dict-object/
        - https://stackoverflow.com/questions/1305532/convert-python-dict-to-object

    >>> o = objectview({'a': 1, 'b': 2})
    >>> o.a, o.b
    (1, 2)
    >>> o = objectview(a=1, b=2)
    >>> o.a, o.b
    (1, 2)
    """
    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        self.__dict__ = d
