from heart.log import setup
from heart.utils.core_helper import setup_constants, setup_extra_constants, setup_globals

#  ╭────────────────────────────────────────────────────────────────────╮
#  │             Set up all global constants using hc class             │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯
setup_constants()
setup_extra_constants()
setup_globals()
#  ╭────────────────────────────────────────────────────────────────────╮
#  │                            setup logger                            │
#  ╰────────────────────────────────────────────────────────────────────╯
setup()
