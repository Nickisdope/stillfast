import sys
sys.path.append('/srv/beegfs02/scratch/gaze_pred/data/xiang/stillfast')

from stillfast.datasets import Ego4dShortTermAnticipationStillVideo
from main import parse_args, load_config
from argparse import Namespace

args = Namespace(cfg_file='configs/sta/STILL_FAST_R50_X3DM_EGO4D_v1_debug.yaml',
            checkpoint=None,
            exp='trial',
            fast_dev_run=True,
            num_shards=1,
            opts=[],
            parallel_test=False,
            test=False,
            test_dir=None,
            train=True,
            val=False)
cfg = load_config(args)
split = 'train'

dataset = Ego4dShortTermAnticipationStillVideo(cfg, split)
print(len(dataset))
dataset[0]