import torch
import torch.nn as nn


import torch.nn as nn

class BaseForecast(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rnn ,num_layers: int = 2):
        super().__init__()
        self.lstm = rnn(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(
            self,
            inp,
            future = [],
            src = None,
            teacher_forcing_ratio = 0.5
        ):
        # inp: [batch, seq_len, input_size]
        preds = []
        out, (hidden) = self.lstm(inp)
        out = out[:, -1, :]
        preds.append(self.fc(out))
        for i in range(len(future[-1])-1):
            if src is not None:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                last_target = src[:,i].unsqueeze(1) if teacher_force else preds[-1]
            else:
                last_target = preds[-1]

            cur_inp = torch.cat([last_target,future[:,i,:]], dim=1).unsqueeze(1)
            out, (hidden) = self.lstm(cur_inp, (hidden))
            out = out[:, -1, :]
            preds.append(self.fc(out))

        return torch.cat(preds, dim=1)

class LSTMForecast(BaseForecast):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__(input_size, hidden_size, nn.LSTM, num_layers)

class GRUForecast(BaseForecast):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__(input_size, hidden_size, nn.GRU, num_layers)


class Seq2SeqForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, time_feature_size,):
        super(Seq2SeqForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Энкодер 
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
         
        )

        self.dec0 = nn.Linear(hidden_size, hidden_size)

        # Декодер принимает предсказанное значение объема + временные признаки
        self.decoder = nn.GRUCell(
            input_size=1 + time_feature_size, 
            hidden_size=hidden_size,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, src, future_time_features, targets=None, teacher_forcing_ratio=0.5):

        batch_size = src.size(0)
        horizon = future_time_features.size(1)

        states, _ = self.encoder(src)

        outputs = torch.zeros(batch_size, horizon, 1, device=src.device)

        last_value = src[:, -1, 0].unsqueeze(1)  # [batch_size, 1]
        hidden = states[:,-1,:]
        hidden = self.dec0(hidden)


        # Декодирование
        for t in range(horizon):
            current_time_features = future_time_features[:, t, :]  # [batch_size, time_feature_size]

            decoder_input = torch.cat([last_value, current_time_features], dim=1) # [batch_size, 1, 1+time_feature_size]

            # Передаем через декодер
            hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(hidden)  # [batch_size, 1]
            outputs[:, t] = prediction

            if targets is not None and t < horizon-1:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                last_value = targets[:, t].unsqueeze(1) if teacher_force else prediction
            else:
                last_value = prediction

        return outputs


class AttentionSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, time_feature_size,):
        super(AttentionSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Энкодер 
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        
        )

        self.dec0 = nn.Linear(hidden_size, hidden_size)

        # Декодер 
        self.decoder = nn.GRUCell(
            input_size=1 + time_feature_size + hidden_size,  
            hidden_size=hidden_size ,
        )

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, src, future_time_features, targets=None, teacher_forcing_ratio=0.5):

        batch_size = src.size(0)
        horizon = future_time_features.size(1)

        states, _ = self.encoder(src)

        outputs = torch.zeros(batch_size, horizon, 1, device=src.device)

        last_value = src[:, -1, 0].unsqueeze(1)  # [batch_size, 1]
        hidden = states[:,-1,:]
        hidden = self.dec0(hidden)


        # Декодирование
        for t in range(horizon):
            current_time_features = future_time_features[:, t, :]  # [batch_size, time_feature_size]

            attention, _ = self.attention(hidden.unsqueeze(1), states, states)
    
            decoder_input = torch.cat([last_value, current_time_features, attention.squeeze(1)], dim=1)# [batch_size, 1, 1+time_feature_size]
            # Передаем через декодер
            hidden = self.decoder(decoder_input, hidden)
            
            prediction = self.fc(hidden)  # [batch_size, 1]
            outputs[:, t] = prediction

            if targets is not None and t < horizon-1:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                last_value = targets[:, t].unsqueeze(1) if teacher_force else prediction
            else:
                last_value = prediction

        return outputs