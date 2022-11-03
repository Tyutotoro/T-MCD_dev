from utils.custom import CustomConstantTimeDataset_new
from utils.custom_constant_time_dataset import CustomConstantTimeDataset
from utils.custom_dataset import CustomDataset
from utils.custom_random_time_dataset import CustomRandomTimeDataset
from utils.image_transform import ImageTransform


def get_datasets(
        csv_filepaths_train :str,
        csv_filepaths_val :str,
        csv_filepaths_test :str,
        img_suffix : str,
        label_suffix :str,
        mode: str,
        datasets: dict,
        phases: list,
        image_size: int,
        augment: bool,
        dataset_type='none',
        train_n=1,
        n_range=None,
        augment_identical=True,
):
    """
    :param phases:
    :param mode:
    :param datasets:
    :param image_size:
    :param augment:
    :param dataset_type:
    :param train_n:
    :param n_range:
    :param augment_identical:
    :return:
    """
    print(mode)
    print(datasets)    
    dataset = datasets[mode]
    if dataset_type == 'none':
        train_dataset = CustomDataset(
            dataset_root=dataset[0],
            transform=ImageTransform(image_size, augment),
            phase=phases[0],
        )
        val_dataset = CustomDataset(
            dataset_root=dataset[1],
            transform=ImageTransform(image_size, augment),
            phase=phases[1],
        )
        test_dataset = CustomDataset(
            dataset_root=dataset[2],
            transform=ImageTransform(image_size, augment),
            phase=phases[2],
        )
    else:
        if dataset_type == 'random':
            train_dataset = CustomRandomTimeDataset(
                dataset_root=dataset[0],
                transform=ImageTransform(image_size, augment),
                phase=phases[0],
            )
        elif dataset_type == 'n_random':
            train_dataset = CustomRandomTimeDataset(
                dataset_root=dataset[0],
                transform=ImageTransform(image_size, augment),
                phase=phases[0],
                n_range=n_range,
            )
        else:
            train_dataset = CustomConstantTimeDataset_new(
                csv_filepath=csv_filepaths_train,
                image_suffix=img_suffix,
                label_suffix=label_suffix,
                image_root=dataset[0],
                label_root=dataset[1],
                transform=ImageTransform(image_size, augment, augment_identical),
                phase=phases[0],
            )
        #     train_dataset = CustomConstantTimeDataset(
        #         dataset_root = dataset[0],
        #         transform=ImageTransform(image_size, augment, augment_identical),
        #         phase=phases[0],
        #         n=0 if dataset_type == 'same' else train_n,
        #     )
        # val_dataset = CustomConstantTimeDataset(
        #     dataset_root = dataset[1],
        #     transform=ImageTransform(image_size, augment, augment_identical),
        #     phase=phases[1],
        #     n=0 if dataset_type == 'same' else train_n,
        #     )
        # test_dataset = CustomConstantTimeDataset(
        #     dataset_root = dataset[2],
        #     transform=ImageTransform(image_size, augment, augment_identical),
        #     phase=phases[2],
        #     n=0 if dataset_type == 'same' else train_n,
        #     )    
        val_dataset  = CustomConstantTimeDataset_new(
                csv_filepath=csv_filepaths_val,
                image_suffix=img_suffix,
                label_suffix=label_suffix,
                image_root=dataset[2],
                label_root=dataset[3],
                transform=ImageTransform(image_size, augment, augment_identical),
                phase=phases[1],
            )
        test_dataset= CustomConstantTimeDataset_new(
                csv_filepath=csv_filepaths_test,
                image_suffix=img_suffix,
                label_suffix=label_suffix,
                image_root=dataset[4],
                label_root=dataset[5],
                transform=ImageTransform(image_size, augment, augment_identical),
                phase=phases[2],
            )
    return train_dataset, val_dataset, test_dataset
