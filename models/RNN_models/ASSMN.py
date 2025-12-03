import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecMN(nn.Module):
    def __init__(self, spec_band, num_classes=128, oly_se=True, init_weights=True):
        super(SpecMN, self).__init__()

        self.oly_se = oly_se
        self.spec_band = spec_band

        # LSTM layers for different parts of the spectrogram
        self.lstm1 = nn.LSTM(input_size=spec_band // 8, hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=spec_band // 4, hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=spec_band // 2, hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=spec_band, hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)

        self.fc = nn.Linear(in_features=128, out_features=128)
        # self.pre = nn.Linear(in_features=128, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec):
        # channel of spec
        d=x_spec.shape[-1]
        # prepare data
        p1_length=int(d/8)
        p2_length=int(d/4)
        p3_length=int(d/2)
        x1=torch.zeros(x_spec.shape[0],8,p1_length)
        x2=torch.zeros(x_spec.shape[0],4,p2_length)
        x3=torch.zeros(x_spec.shape[0],2,p3_length)
        x4=x_spec.reshape(x_spec.shape[0],1,x_spec.shape[-1])
        ## eight parts

        ## eight parts
        start=0
        end = min(start + p1_length, d)
        for i in range(8):
            x1[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p1_length, d)
        ## four parts
        start = 0
        end = min(start + p2_length, d)
        for i in range(4):
            x2[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p2_length, d)
        ## two parts
        start = 0
        end = min(start + p3_length, d)
        for i in range(2):
            x3[:, i, :] = x_spec[:,start:end]
            start = end
            end = min(start + p3_length, d)
        # Pass the parts through respective LSTM layers
        _, (y_1, _) = self.lstm1(x1)
        _, (y_2, _) = self.lstm2(x2)
        _, (y_3, _) = self.lstm3(x3)
        _, (y_4, _) = self.lstm4(x4)

        # Squeeze the LSTM outputs and sum them
        y_1 = y_1.squeeze(0)
        y_2 = y_2.squeeze(0)
        y_3 = y_3.squeeze(0)
        y_4 = y_4.squeeze(0)
        y = y_1 + y_2 + y_3 + y_4

        # Apply fully connected layer and ReLU activation
        y = F.relu(self.fc(y))

        # if self.oly_se:
        #     score = self.pre(y)
        #     return score

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)




# Testing the model
if __name__ == '__main__':
    input = torch.randn(2, 204) # Random input spectrogram

    model = SpecMN(spec_band=204, num_classes=128)
    # model = specMN_scheme2(spec_band=204, num_classes=7)

    output = model(input)
    print(output.shape)