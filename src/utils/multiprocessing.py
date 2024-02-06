import torch.multiprocessing as mp

manager = None


def get_list_cls():
    global manager
    if manager is None:
        manager = mp.Manager()
    return manager.list


def get_dict_cls():
    global manager
    if manager is None:
        manager = mp.Manager()
    return manager.dict
