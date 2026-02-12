#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
from numpy.ctypeslib import ndpointer

def get_backproj_RB_ctypes():
    dll = ctypes.CDLL('./BackProjRB_GPU2D_core.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.BackProjGPU
    func.restype = None

    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int]
    return func
