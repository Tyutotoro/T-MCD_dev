from typing import Dict, List

datasets: Dict[str, List[str]] = {
    "dbscreen": [
    #     "./data/dbscreen/interval_40s/train",
    #     "./data/dbscreen/interval_40s/val",
    #     "./data/dbscreen/interval_40s/test",
        
        "./data/dbscreen/interval_40s/train/image",
        "./data/dbscreen/interval_40s/train/label",
        "./data/dbscreen/interval_40s/val/image",
        "./data/dbscreen/interval_40s/val/label",
        "./data/dbscreen/interval_40s/test/image",
        "./data/dbscreen/interval_40s/test/label",
    ],
    "wddd": [
        "./data/wddd/interval_40s/train",
        "./data/wddd/interval_40s/val",
        "./data/wddd/interval_40s/test",
    ],
    "atp-4": [
        "./data/atp-4/atp_data/train/image",
        "",
        "./data/atp-4/atp_data/val/image",
        "",
        "./data/atp-4/atp_data/test/image",
        "",
    ],
    "WDDD2_WT": [
        "./data/WDDD2_WT/train/image",
        "",
        "./data/WDDD2_WT/val/image",
        "",
        "./data/WDDD2_WT/test/image",
        "",
    ]

}

train_count = 1

seed = None
batch_size = 5
num_iter = 8
num_epochs = 500  #500が基本

# params for dataset and data loader
source = "dbscreen"                     # option: (dbscreen, wddd)
target = "atp-4"                         # option: (dbscreen, wddd)
dataset_type = 'constant'               # option: (constant, randam, same, none)
val_index_list = range(41)[1:]

# params for setting up models
model_name = "co_detection_cnn"         # option: (co_detection_cnn, unet)  *If you choose "unet", set "dataset_type" to "none".
model_g_filename = 'model_g.pt'
model_f1_filename = 'model_f1.pt'
model_f2_filename = 'model_f2.pt'
res = '50'

# params for training network
input_channel = 1
num_class = 2
image_size = 256
up_mode = 'upsample'
junction_point = 1

# params for optimizing models
learning_rate = 1e-3
num_k = 3
weight_decay = 5e-4
momentum = 0.5

max_discrepancy = False
augment = False

train_n = 1
n_range = None
augment_identical = True

palette = [
    [0, 0, 0],
    [255, 255, 255],
]
