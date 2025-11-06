# -*- encoding: utf-8 -*-
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.data.dem_dataset import DemDataset


def create_dataset(args, path:str, mode:str) :
    """init dataset

    Args:
        args (_type_): parameters
        path (str): data path (.txt)
        mode (str): train/val

    Returns:
        tuple: prefetcher, num of images, iterations of every epoch
    """
    dataset = DemDataset(data_path=path, mode=mode, crop_size=args.patch_size, scale=args.scale)
    batch_size = args.batch_size if mode == 'train' else args.test_batch_size
    sampler = None
    if mode == 'train':
        sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=args.workers,
                            drop_last=True, pin_memory=False, sampler=sampler)
    num_image = dataset.__len__()
    return dataloader, num_image