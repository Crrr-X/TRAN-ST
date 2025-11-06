import argparse

parser = argparse.ArgumentParser(description='unsupervised super-resolution')


# Hardware specifications
parser.add_argument('--workers', type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument('--n_GPUs', type=int, default=4,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--gpu-ids', type=list,
                    default=[0,1,2,3], help="use which gpu in environ to train")

# Data specifications

parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--resolution', type=int, default=30,
                    help='the resolution of DEM')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=256,
                    help='maximum value of RGB')
parser.add_argument('--n_channels', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--dataset_dir', type=str,
                    default="SD_SAR")
parser.add_argument('--mean', type=float, default=0.,
                    help='maximum value of RGB')
parser.add_argument('--std', type=float, default=0.,
                    help='maximum value of RGB')

# Model specifications
parser.add_argument('--model_name', type=str, required=True,
                    help='model name')
parser.add_argument('--model_type', type=str, default="supervised",
                    help='supervised or unsupervised model')
parser.add_argument('--pretrained-path', type=str,
                    default=None)
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_res_blocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_features', type=int, default=256,
                    help='number of feature maps')
parser.add_argument('--isTrain', action="store_false",
                    default=True, help='is test or not')
parser.add_argument('--resume', type=str, default=None, help='checkpoint name')
# DRN model config
parser.add_argument('--n_blocks', type=int, default=40,
                    help="number of DRN blocks")
parser.add_argument('--n_feats', type=int, default=20,
                    help='channels of DRN features ')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')

# pretrained model
parser.add_argument('--resume_DT', type=str, default=None, help='domain transfer checkpoint name')
parser.add_argument('--resume_SR', type=str, default=None, help='SR checkpoint name')

# Training specifications

parser.add_argument('--epochs', type=int, default=1200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, 
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=8, 
                    help='input batch size for val')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, 
                    help='learning rate')
parser.add_argument('--milestones', type=list, default=[400, 600, 800],
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--weight', type=list, default=[10,0,0],
                    help='loss function weight')

# DPP
parser.add_argument("--local_rank", type=int, default=0)

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
