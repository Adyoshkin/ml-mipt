import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

model_cnn = nn.Sequential()

model_cnn.add_module('conv_1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
model_cnn.add_module('relu_1', nn.ReLU())

model_cnn.add_module('conv_2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
model_cnn.add_module('relu_2', nn.ReLU())

model_cnn.add_module('conv_3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model_cnn.add_module('relu_3', nn.ReLU())
model_cnn.add_module('max_pool_3', nn.MaxPool2d(kernel_size=2, stride=2))

model_cnn.add_module('flat', Flatten())

model_cnn.add_module('fc_1',nn.Linear(in_features=64*14*14, out_features=128))
model_cnn.add_module('relu_fc_1', nn.ReLU())

model_cnn.add_module('fc_2',nn.Linear(in_features=128, out_features=10))

###############################################################################################################################

model_cnn_overfit = nn.Sequential()

model_cnn_overfit.add_module('conv_1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
model_cnn_overfit.add_module('relu_1', nn.ReLU())

model_cnn_overfit.add_module('conv_2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
model_cnn_overfit.add_module('relu_2', nn.ReLU())

model_cnn_overfit.add_module('conv_3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model_cnn_overfit.add_module('relu_3', nn.ReLU())

model_cnn_overfit.add_module('conv_4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
model_cnn_overfit.add_module('relu_4', nn.ReLU())
model_cnn_overfit.add_module('max_pool_4', nn.MaxPool2d(kernel_size=2, stride=2))

model_cnn_overfit.add_module('flat', Flatten())

model_cnn_overfit.add_module('fc_1',nn.Linear(in_features=64*14*14, out_features=256))
model_cnn_overfit.add_module('relu_fc_1', nn.ReLU())

model_cnn_overfit.add_module('fc_2',nn.Linear(in_features=256, out_features=10))

###############################################################################################################################

model_cnn_overfit_reg = nn.Sequential()

model_cnn_overfit_reg.add_module('conv_1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
model_cnn_overfit_reg.add_module('bn_1',nn.BatchNorm2d(16))
model_cnn_overfit_reg.add_module('relu_1', nn.ReLU())

model_cnn_overfit_reg.add_module('conv_2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
model_cnn_overfit_reg.add_module('bn_2',nn.BatchNorm2d(32))
model_cnn_overfit_reg.add_module('relu_2', nn.ReLU())

model_cnn_overfit_reg.add_module('conv_3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model_cnn_overfit_reg.add_module('bn_3',nn.BatchNorm2d(64))
model_cnn_overfit_reg.add_module('relu_3', nn.ReLU())

model_cnn_overfit_reg.add_module('dropout_1', nn.Dropout(0.5))

model_cnn_overfit_reg.add_module('conv_4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
model_cnn_overfit_reg.add_module('bn_4',nn.BatchNorm2d(64))
model_cnn_overfit_reg.add_module('relu_4', nn.ReLU())
model_cnn_overfit_reg.add_module('max_pool_4', nn.MaxPool2d(kernel_size=2, stride=2))

model_cnn_overfit_reg.add_module('dropout_2', nn.Dropout(0.5))

model_cnn_overfit_reg.add_module('flat', Flatten())

model_cnn_overfit_reg.add_module('fc_1',nn.Linear(in_features=64*14*14, out_features=256))
model_cnn_overfit_reg.add_module('relu_fc_1', nn.ReLU())

model_cnn_overfit_reg.add_module('dropout_3', nn.Dropout(0.3))

model_cnn_overfit_reg.add_module('fc_2',nn.Linear(in_features=256, out_features=10))