#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File Name: log.py
import logging
from typing import Optional, cast

from rich.logging import RichHandler

TRACE_LEVEL = 5
TRACE_LOGGERS = [f"!{__name__}"]


class CustomLogger(logging.Logger):
    """Custom Logger, initialized with rich handler"""

    def __init__(self, name: Optional[str], level: logging = logging.NOTSET):
        super().__init__(name, level)
        self.addHandler(RichHandler())

    def trace(self, msg: Optional[str], *args, **kwargs) -> None:
        """Trace Level message Custom for this logger"""
        if self.isEnabledFor(TRACE_LEVEL):
            self.log(TRACE_LEVEL, msg, *args, **kwargs)


def get_logger(name: Optional[str] = None) -> CustomLogger:
    """Return a logger with the given name. Which tends to be done by
    get_logger(__name__) in most cases.

    Parameters
    ----------
    name : Optional[str]
        Name of instance / module

    Returns
    -------
    CustomLoggerLogger
        CustomerLoger
    """
    return cast(CustomLogger, logging.getLogger(name))


def setup() -> None:
    """ setup file for logger - initalises level, format  and its own trace """
    logging.TRACE = TRACE_LEVEL
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    logging.setLoggerClass(CustomLogger)

    root_log = get_logger()

    format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    logging.Formatter(format_string)
    root_log.setLevel(logging.INFO)
    _set_trace_loggers()


def _set_trace_loggers() -> None:
    """
    Set loggers to the trace level according to the value from the BOT_TRACE_LOGGERS env var.
    Options is we have a  list[str] where str starts with either ! or * to indicate logger
    if ! then we set the logger to the trace so
    test = ["!", "... "] will all be set to trace
    test = ["*", "..."] will all be set to debug
    """
    trace_loggers = TRACE_LOGGERS
    if not trace_loggers:
        return
    for logger_name in trace_loggers:
        if logger_name.startswith("!"):
            logger_name = logger_name.lstrip("!")
            get_logger(logger_name).setLevel(TRACE_LEVEL)
        elif logger_name.startswith("*"):
            logger_name = logger_name[1:]
            get_logger(logger_name).setLevel(logging.DEBUG)
        else:
            get_logger(logger_name).setLevel(TRACE_LEVEL)
        get_logger(logger_name).trace(get_logger(logger_name).level)
