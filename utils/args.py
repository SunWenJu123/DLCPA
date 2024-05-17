from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='deep inc params', allow_abbrev=False)
    parser.add_argument('--transform', type=str, default='default',
                        help='default or pytorch.')
    parser.add_argument('--featureNet', type=str, default=None,
                        help='feature extractor')
    parser.add_argument('--img_dir', type=str, default='img/',
                        help='image dir')
    parser.add_argument('--nt', type=int, default=None,
                        help='task number')
    parser.add_argument('--t_c_arr', type=str, default=None,
                        help='class array for each task')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed if None')
    parser.add_argument('--validation', type=bool, default=False,
                        help='is test with the validation set')
    parser.add_argument('--class_shuffle', type=bool, default=False,
                        help='is random shuffle the classes order')
    parser.add_argument('--backbone', type=str, default='None',
                        help='is test with the validation set')
    parser.add_argument('--root', type=str, default='./data/',
                        help='dictionary of dataset')
    args = parser.parse_known_args()[0]
    return args