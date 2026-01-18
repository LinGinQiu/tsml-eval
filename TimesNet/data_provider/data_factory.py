from functools import partial

from torch.utils.data import DataLoader

from TimesNet.data_provider.data_loader import UEAloader
from TimesNet.data_provider.uea import collate_fn


def data_provider(args, data_zip, flag):
    Data = UEAloader
    if flag.lower() == "test":
        shuffle_flag, drop_last = False, False
    else:
        shuffle_flag, drop_last = True, True
    batch_size = args.batch_size

    if args.task_name == "classification":
        data_set = Data(
            args=args,
            data_zip=data_zip,
            flag=flag,
        )
        collate_with_maxlen = partial(collate_fn, max_len=args.seq_len)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_with_maxlen,
        )
        return data_set, data_loader
    else:
        raise NotImplementedError
