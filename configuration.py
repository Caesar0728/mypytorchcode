from pathlib import Path

def get_config():

    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": 'en',
        "lang_tgt": 'it',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    # 这里的config['model_basename'] 是tmodel_，
    # 模型没跑完一个epoch就会存一个记录，比如第一个epoch跑完就会有tmodel_0.pt
    # 比如第2个epoch跑完就会有tmodel_1.pt, 如此类推
    model_filename = f"{config['model_basename']}{epoch}.pt"  # tmodel_ .pt"

    # 然后这里返回的是一个完整的路径string
    # './{config['datasource']}_{config['model_folder']/tmodel_0.pt
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    # 通过get_weights_file_path这个函数可以知道， 没跑完一个epoch就会生成一个tmodel_{epoch}.pt的文件，
    # 然后latest_weights_file_path这个函数是希望在这一堆的tmodel_{epoch}.pt epoch = 0， 1， 2， 3。。。文件中找到最后一个
    # 即最新的一个

    # 注意这里的model_filename = f"{config['model_basename']}*" = "tmodel_*"
    # 下面Path(model_folder).glob(model_filename))是指在model_folder的路径下（文件夹中）找到所有"tmodel_"开头的文件
    # 并且将他们变成列表，
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    # 对这个列表排序， 然后取这个列表的最后一个元素， 就是最新的latest的文件了
    weights_files.sort()
    return str(weights_files[-1])