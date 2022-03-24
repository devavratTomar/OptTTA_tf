import os
import numpy as np
import re

COLORS = np.array([[0, 0, 0], [197,17,17], [239,125,14], [246,246,88], [237,84,186], [113,73,30]], dtype=np.uint8)

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def color_seg(seg):
    c_seg = COLORS[seg]
    return c_seg

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    new_items = items.copy()
    new_items.sort(key=natural_keys)
    return new_items

class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def moving_average(self, old, new):
        s = 0.98
        return old * (s) + new * (1 - s)

    def update_metrics(self, metric_dict, smoothe=True):
        for k, v in metric_dict.items():
            if k in self.metrics and smoothe:
                self.metrics[k] = self.moving_average(self.metrics[k], v)
            else:
                self.metrics[k] = v

    def current_metrics(self):
        return self.metrics


def overlay_segs(img, seg, alpha=0.2):
    """
    imgs should be in range [-1, 1] and shape  H x W x C
    seg should have integer range with same spatial shape as H x W.
    """
    print(img.shape, seg.shape)
    assert img.shape[:2] == seg.shape[:2]
    mask = (seg != 0)[..., np.newaxis]

    color_seg = COLORS[seg]/255.0
    img = np.clip(img, -1, 1) * 0.5 + 0.5
    
    merged = mask*(alpha*color_seg + (1-alpha)*img) + (~mask) * img

    return merged 