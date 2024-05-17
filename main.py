import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils.args import get_args
from utils.training import train_il
from utils.conf import set_random_seed
import torch




def main():
    args = get_args()
    args.model = 'dlcpa'
    args.seed = None
    args.validation = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        set_random_seed(args.seed)


    # CIFAR-100
    args.dataset = 'seq-cifar100'
    args.print_freq = 10
    args.n_epochs = 100
    args.classifier = 'linear'
    args.scheduler_step = 99
    args.ssl_leaner = 'moco'

    args.lr = 0.005
    args.clslr = 0.2
    args.batch_size = 32
    args.ssl_weight = 1
    args.sl_weight = 10
    args.weight_decay = 0
    args.momentum = 0
    args.task1weight = 1

    # seq-tinyimg
#    args.dataset = 'seq-tinyimg'
#    args.print_freq = 10
#    args.n_epochs = 100
#    args.classifier = 'linear'
#    args.scheduler_step = 90
#    args.ssl_leaner = 'moco'
#
#    args.lr = 0.02
#    args.clslr = 0.05
#    args.batch_size = 512
#    args.ssl_weight = 1
#    args.sl_weight = 10
#    args.weight_decay = 0
#    args.momentum = 0


    for conf in [1]:
        print("")
        print("=================================================================")
        print("==========================", "index", ":", conf, "==========================")
        print("=================================================================")
        print("")
        train_il(args)



if __name__ == '__main__':
    main()
