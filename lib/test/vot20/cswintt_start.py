import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from lib.test.vot20.cswintt_vot20 import run_vot_exp
run_vot_exp()