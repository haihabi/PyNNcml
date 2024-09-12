import numpy as np

from pynncml.datasets.dataset import LinkDataset, SubSequentLinkDataset


def linkdataset2subsequent(in_linkdataset: LinkDataset, subsequent_size=128, threshold=0.1):
    ref_list = []
    data_list = []
    meta_list = []
    for i in range(len(in_linkdataset)):
        gauge, rsl, tsl, meta = in_linkdataset[i]
        for j in range(rsl.shape[0] - subsequent_size):
            if gauge[j + subsequent_size - 1] > threshold:
                ref = gauge[j + subsequent_size - 1]
                if ref == 0:
                    raise Exception()
                _rsl = rsl[j:j + subsequent_size]
                _tsl = tsl[j:j + subsequent_size]
                meta_list.append(meta)
                ref_list.append(ref)
                data_list.append(np.concatenate([_rsl, _tsl], axis=-1))
    return SubSequentLinkDataset(data_list, ref_list, meta_list)