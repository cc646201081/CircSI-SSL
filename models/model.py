from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        hidden_channel = 128
        # data1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, hidden_channel, kernel_size=1,
                      stride=configs.stride, bias=False, padding=0),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

        # data2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(101, hidden_channel, kernel_size=4,
                      stride=configs.stride, bias=False, padding=0),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3,
                  stride=configs.stride, bias=False, padding=0),
            nn.ReLU(),
        )



        # data3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(99, hidden_channel, kernel_size=1,
                      stride=configs.stride, bias=False, padding=0),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in, tag):
        if tag==1:
            x = self.conv_block1(x_in)

        elif tag==2:
            x = self.conv_block2(x_in)

        elif tag==3:
            x = self.conv_block3(x_in)

        return  x
