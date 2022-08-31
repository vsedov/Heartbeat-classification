class H:

    def __init__(self):
        ...


h = H()


def h_regeister(f):

    def wrapped(*args, **kwargs):
        h.__dict__[f.__name__] = f
        return f(*args, **kwargs)
