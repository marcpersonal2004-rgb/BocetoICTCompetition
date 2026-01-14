import mindspore.nn as nn
import mindspore.ops as ops

class RiverCNN(nn.Cell):
    def __init__(self, num_classes):
        super(RiverCNN, self).__init__()

        self.features = nn.SequentialCell([
            nn.Conv2d(3, 32, kernel_size=3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, pad_mode='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])

        self.flatten = ops.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dense(256, num_classes)
        ])

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
