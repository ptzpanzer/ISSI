import sys
import json
from Datasets import aio_dataloader_lspe as dl


if __name__ == '__main__':
    settings = json.loads(sys.argv[1])

    settings["debug"] = False
    settings["k"] = 5
    settings["pos_enc_dim"] = 32
    settings["pe_init"] = 'rand_walk'
    settings['tmpdir'] = '$Path to your workspace$'

    print(f"Trigger Work {sys.argv[1]}", end="\r", flush=True)

    dataset_train = dl.IntpDataset(settings=settings, mask_distance=settings["mask_distance"], call_name=settings["call_name"])

    print("Work Done!", end="\r", flush=True)
    