"""
A Pdb subclass that may be used from a forked multiprocessing child.
ref: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
"""  # noqa: E501

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child."""

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)
