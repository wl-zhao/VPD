# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

from configs.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--ckpt_dir',   type=str,
                    default='./ckpt/best_model_nyu.ckpt', 
                    help='load ckpt path')
        parser.add_argument('--crop_h',  type=int, default=448)
        parser.add_argument('--crop_w',  type=int, default=576)       
        
        return parser


