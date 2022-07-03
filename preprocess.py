import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
import mindspore.dataset as ds

def create_dataset(data_path, mean=None, std=None, repeat_num=1, batch_size=32, usage='train'):
    """
    数据处理

    Args:
        dataset_path (str): 数据路径
        repeat_num (int): 数据重复次数
        batch_size (int): 批量大小
        usage (str): 训练或测试

    Returns:
        Dataset对象
    """

    # 载入数据集
    if usage=='train':
        data = ds.ImageFolderDataset(data_path)
    else:
        # 每类取63个样本（最小类样本总数）
        sample_num = 63
        data = ds.ImageFolderDataset(data_path, sampler=ds.PKSampler(sample_num))

    # 打乱数据集
    data = data.shuffle(buffer_size=10000)

    # 设定resize和normalize参数
    image_size = 224
    rgb_mean = mean
    rgb_std = std

    # 定义算子
    if usage=='train':
        trans = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Decode(),
            CV.Resize(256),
            CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]

    type_cast_op = C.TypeCast(mstype.int32)


    # 算子运算
    data = data.map(operations=trans, input_columns="image")
    data = data.map(operations=type_cast_op, input_columns="label")


    # 批处理
    if usage == 'train':
        drop_remainder = True
    else:
        drop_remainder = False
        
    data = data.batch(batch_size, drop_remainder=drop_remainder)

    # 重复
    data = data.repeat(repeat_num)

    return data
