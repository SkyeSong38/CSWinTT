from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
from . import vot
import sys
import time
import os
from lib.test.evaluation import Tracker
from lib.test.vot20.vot20_utils import *

class cswintt_vot20(object):
    def __init__(self, tracker_name='cswintt', para_name='baseline_cs'):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "vot20", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgb, mask=None):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        region = mask
        self.H, self.W, _ = img_rgb.shape
        # init_info = {'init_bbox': region}
        init_info = {'init_bbox': np.array([region.x, region.y, region.width, region.height],dtype=np.float)}
        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track
        # return {"target_bbox": self.state,
        #         "conf_score": conf_score}
        outputs = self.tracker.track(img_rgb)
        pred_bbox = outputs['target_bbox']
        pred_bbox = vot.Rectangle(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3])
        # final_mask = mask_from_rect(pred_bbox, (self.W, self.H))
        return pred_bbox

def run_vot_exp(tracker_name="cswintt", para_name="baseline_cs", vis=False):
    torch.set_num_threads(1)
    save_root = ''
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = cswintt_vot20(tracker_name=tracker_name, para_name=para_name)
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    # mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, selection)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1 = tracker.track(image)
        handle.report(b1)