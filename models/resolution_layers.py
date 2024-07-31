

class ResolutionLayer(object):
    def __init__(self) -> None:
        pass

class ResolutionWithConv(ResolutionLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False) -> None:
        super(ResolutionWithConv, self).__init__()
        pass

    def forward(self, x):
        return self.relu(self.conv(x))