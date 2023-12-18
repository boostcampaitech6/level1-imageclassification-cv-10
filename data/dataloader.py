import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler

def create_data_loader(dataset, batch_size, use_cuda, sampler=None, collate=None, is_train=True):
    """
    지정된 데이터셋과 설정에 따라 DataLoader를 생성합니다.

    Args:
        dataset: 사용할 데이터셋.
        batch_size (int): 배치 크기.
        use_cuda (bool): CUDA 사용 여부.
        sampler (str, optional): 사용할 샘플러. 'ImbalancedSampler' 또는 'WeightedSampler' 등이 될 수 있음.
        collate (callable, optional): 배치에 적용할 collate 함수.
        is_train (bool, optional): 훈련 데이터셋인지 여부. 기본값은 True.

    Returns:
        DataLoader: 생성된 DataLoader 객체.
    """
    if sampler is None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            shuffle=is_train,
            pin_memory=use_cuda,
            drop_last=is_train
        )
    elif sampler == "ImbalancedSampler":
        labels = [dataset[i][1] for i in range(len(dataset))]
        loader = DataLoader(
            dataset,
            sampler=ImbalancedDatasetSampler(dataset, labels=labels),
            batch_size=batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            pin_memory=use_cuda,
            drop_last=is_train
        )
    elif sampler == "WeightedSampler":
        BASE_WEIGHT = [6.885245901639344,
                       9.21951219512195,
                       45.54216867469879,
                       5.163934426229508,
                       4.626682986536108,
                       34.678899082568805,
                       34.42622950819672,
                       46.09756097560975,
                       227.710843373494,
                       25.81967213114754,
                       23.133414932680537,
                       173.39449541284404,
                       34.42622950819672,
                       46.09756097560975,
                       227.710843373494,
                       25.81967213114754,
                       23.133414932680537,
                       173.39449541284404]
        weights = [BASE_WEIGHT[dataset[i][1]] for i in range(len(dataset))]
        weightedsampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
        loader = DataLoader(
            dataset,
            sampler=weightedsampler,
            batch_size=batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            pin_memory=use_cuda,
            drop_last=is_train
        )

    return loader