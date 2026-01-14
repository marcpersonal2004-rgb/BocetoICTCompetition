import mindspore as ms
from mindspore import nn, Model
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig

from model import RiverCNN
from dataset import create_dataset
from config import NUM_CLASSES, LR, EPOCHS

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

train_path = "/cache/data/train"
val_path = "/cache/data/val"

train_dataset = create_dataset(train_path)
val_dataset = create_dataset(val_path, training=False)

network = RiverCNN(NUM_CLASSES)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(network.trainable_params(), learning_rate=LR)

model = Model(network, loss_fn=loss, optimizer=optimizer, metrics={"Accuracy": nn.Accuracy()})

ckpt_config = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpt_cb = ModelCheckpoint(prefix="river_cnn", directory="/cache/ckpt", config=ckpt_config)

model.train(
    EPOCHS,
    train_dataset,
    callbacks=[LossMonitor(), ckpt_cb],
    dataset_sink_mode=True
)

acc = model.eval(val_dataset)
print("Accuracy:", acc)
