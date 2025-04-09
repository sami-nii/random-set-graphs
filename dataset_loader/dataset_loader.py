from dataset_loader.loader_cora import loader_cora
from dataset_loader.loader_ogb_arxiv_year import load_ogb_arxiv_year
from dataset_loader.loader_squirrel import laoder_squirrel
from dataset_loader.loader_chameleon import laoder_chameleon


DATASET_STORAGE_PATH = "./dataset/"


def dataset_loader(dataset_name: str, config):
    if dataset_name == 'cora':
        return loader_cora(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'ogb_arxiv_year': # TODO fix the seed for the splits
        return load_ogb_arxiv_year(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'squirrel':
        return laoder_squirrel(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'chameleon':
        return laoder_chameleon(DATASET_STORAGE_PATH, config)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
