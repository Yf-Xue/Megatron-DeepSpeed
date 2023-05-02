import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--masked_softmax_fusion', action='store_true')
args = parser.parse_args(['--masked_softmax_fusion'])
args.rank = 0

import megatron
megatron.fused_kernels.load(args)
