#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX 
import os
from jinja2 import evalcontextfilter, filters
import numpy as np
from PIL import Image

def remove_file(path):
    try:
        os.remove(path)
    except:
        pass


@evalcontextfilter
def is_maximum_element(eval_ctx, value, a_list):
    return (value == max(a_list))


def gray2bw(pil_img):
    img = Image.fromarray(np.uint8(pil_img * 16))
    bw = img.point(lambda x: 0 if x < 150 else 255)
    return np.array(np.uint8(bw))

binarize = np.vectorize(gray2bw)


filters.FILTERS['is_maximum_element'] = is_maximum_element