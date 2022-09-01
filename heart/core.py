class HC:

    def __init__(self):
        ...


hc = HC()


def hc_register(f):
    hc.__dict__[f.__name__] = f
    return f


def hc_deco(f):

    def wrapped(*args, **kwargs):
        hc.__dict__[f.__name__] = f
        return f(*args, **kwargs)

    return wrapped


def hc_register_const(name, value):
    hc.__dict__[name] = value
    return value


class HelperFunctions:

    def __init__(self):
        ...


hp = HelperFunctions()


def hp_register(f):
    hp.__dict__[f.__name__] = f
    return f
