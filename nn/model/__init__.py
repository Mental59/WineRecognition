from typing import Optional, List

import torch
from torch import nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, embedding_dim: int, hidden_dim: int,
                 padding_idx: int = None, custom_features_size: int = 0):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.hidden2tags = nn.Linear(hidden_dim * 2 + custom_features_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def get_lstm_features(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return x

    def neg_log_likelihood(
            self,
            x: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.BoolTensor] = None,
            custom_features: Optional[torch.Tensor] = None,
            reduction: str = 'sum'
    ) -> torch.Tensor:
        """
        :param x: IntTensor or LongTensor of arbitrary shape containing the indices to extract
        :param tags: Sequence of tags tensor of size (batch_size, seq_length)
        :param mask: BoolTensor of size (batch_size, seq_length)
        :param custom_features: Our features of shape (batch_size, seq_length, num_custom_features)
        :param reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        :return: `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        x = self.get_lstm_features(x)
        if custom_features is not None and custom_features.size(-1) != 0:
            x = torch.cat((x, custom_features), dim=2)
        x = self.hidden2tags(x)
        x = -self.crf(x, tags, mask, reduction)
        return x

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
            custom_features: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        :param x: IntTensor or LongTensor of arbitrary shape containing the indices to extract
        :param mask: BoolTensor of size (batch_size, seq_length)
        :param custom_features: Our features of shape (batch_size, seq_length, num_custom_features)
        :return: Tensor of size (batch_size, seq_length)
        """
        x = self.get_lstm_features(x)
        if custom_features is not None and custom_features.size(-1) != 0:
            x = torch.cat((x, custom_features), dim=2)
        x = self.hidden2tags(x)
        x = self.crf.decode(x, mask)
        return x


def main():
    tag_to_ix = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    vocab_size = 10
    embedding_dim = 8
    hidden_dim = 6

    model = BiLSTM_CRF(vocab_size, len(tag_to_ix), embedding_dim, hidden_dim)
    input = torch.tensor([
        [0, 5, 2, 1, 4],
        [8, 9, 2, 1, 1],
        [5, 5, 2, 1, 4],
        [7, 2, 1, 7, 1]
    ], dtype=torch.int32)
    tags = torch.LongTensor([
        [0, 2, 2, 3, 4],
        [1, 1, 2, 4, 4],
        [2, 2, 3, 4, 4],
        [1, 2, 1, 3, 4],
    ])
    output = model.neg_log_likelihood(input, tags)
    print(output)


if __name__ == '__main__':
    main()
