#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX 
import os
from jinja2 import evalcontextfilter, filters

def remove_file(path):
    try:
        os.remove(path)
    except:
        pass


@evalcontextfilter
def is_maximum_element(eval_ctx, value, a_list):
    return (value == max(a_list))

filters.FILTERS['is_maximum_element'] = is_maximum_element