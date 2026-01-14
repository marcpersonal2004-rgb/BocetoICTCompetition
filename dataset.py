import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from config import IMAGE_SIZE, BATCH_SIZE

def create_dataset(data_path, training=True):
    dataset = ds.ImageFolderDataset(data_path, shuffle=training)

    transform = [
        vision.Decode(),
        vision.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]

    type_cast = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.map(operations=type_cast, input_columns="label")
    dataset = dataset.batch(BATCH_SIZE)

    return dataset
