def config():
    config = {
        "batch_size": 32,
        "train": '../../data/Reference_ecg/train_ecg.json',
        "dev": '../../data/Reference_ecg/dev_ecg.json',
        "save_dir": "../../data/saved_ecg/"
    }
    return config
