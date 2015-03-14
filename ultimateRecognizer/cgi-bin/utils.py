#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX 
import os


def remove_file(path):
    try:
        os.remove(path)
    except:
        pass