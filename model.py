import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _generate_square_subsequent_mask

import lightning as pl


class LearnablePositionalEncoding(pl.LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        positions = torch.arange(max_len)
        self.register_buffer("positions", positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape (S, d_model)
        encoding = self.embedding(self.positions)
        # shape (1, S, d_model)
        encoding = encoding.unsqueeze(0)
        # x is shape (B, S, d_model) where B is batch size and S is sequence length
        return x + encoding[:, : x.shape[1], :]


class DecoderLayer(pl.LightningModule):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.cross_attn1 = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.cross_attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.gelu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        mem1: torch.Tensor,
        mem2: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = _generate_square_subsequent_mask(tgt.shape[1], device=self.device)
        x = tgt
        # self attention
        x = x + self.dropout1(
            self.self_attn(x, x, x, attn_mask=tgt_mask, is_causal=True)[0]
        )
        x = self.norm1(x)
        # cross attention with previous round predictions
        x = x + self.dropout2(self.cross_attn1(x, mem1, mem1)[0])
        x = self.norm2(x)
        # cross attention with encoder output
        x = x + self.dropout3(self.cross_attn2(x, mem2, mem2)[0])
        x = self.norm3(x)
        # feedforward
        x = x + self.dropout4(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )
        x = self.norm4(x)
        return x


class Decoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        mem1: torch.Tensor,
        mem2: torch.Tensor,
    ) -> torch.Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, mem1, mem2)
        return output


class EncoderLayer(pl.LightningModule):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.gelu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        mask = F._canonical_mask(
            mask=mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        x = self.norm1(x + self._self_attention(x, mask))
        x = self.norm2(x + self._feed_forward(x))
        return x

    def _self_attention(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.dropout1(x)
        return x

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class Encoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, mask)
        return output


class ContModel(pl.LightningModule):
    def __init__(
        self,
        N_D: int,
        N_L: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.N_D = N_D
        self.N_L = N_L
        self.d_model = d_model
        self.detector_embedding = nn.Embedding(2, d_model)
        self.detector_positional_embedding = LearnablePositionalEncoding(
            d_model,
            dropout,
            max_len=N_D,
        )
        self.encoder = Encoder(
            d_model, nhead, dim_feedforward, dropout, num_encoder_layers
        )
        self.error_embedding = nn.Embedding(4, d_model)
        self.error_positional_embedding = LearnablePositionalEncoding(
            d_model,
            dropout,
            max_len=N_L,
        )
        self.decoder = Decoder(
            d_model, nhead, dim_feedforward, dropout, num_decoder_layers
        )
        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None,
        num_ct_rounds: int,
        c: int,
    ) -> torch.Tensor:
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        # src shape (B, N_D * N_R)
        # tgt shape (B, (N_L+1)*N_R)
        num_rounds = src.shape[1] // self.N_D
        all_decoder_outputs = torch.zeros(
            (src.shape[0], self.N_L * (num_rounds - num_ct_rounds)), device=self.device
        )
        cont_thoughts = []
        encoder_output = None
        for i in range(num_rounds):
            if src_mask is not None:
                tmp_src_mask = src_mask[
                    i * self.N_D : (i + 1) * self.N_D, i * self.N_D : (i + 1) * self.N_D
                ]
            else:
                tmp_src_mask = None
            # Pass through encoder
            # (B, N_D)
            dets = src[:, i * self.N_D : (i + 1) * self.N_D]
            # (B, N_D, d_m)
            if i == 0:
                all_zeros = torch.zeros_like(dets, device=self.device)
                encoder_output = self.detector_embedding(all_zeros)
                encoder_output = self.detector_positional_embedding(encoder_output)
            embedded_dets = self.detector_embedding(dets)
            embedded_dets = self.detector_positional_embedding(embedded_dets)
            embedded_dets += encoder_output
            encoder_output = self.encoder(embedded_dets, tmp_src_mask)
            # Pass through decoder
            if i < num_ct_rounds:
                cont_thoughts.append([])
                # "start of thought" token
                # (B, 1)
                sot = (
                    torch.ones((src.shape[0], 1), device=self.device, dtype=torch.int)
                    * 3
                )
                # (B, d_m)
                sot = self.error_embedding(sot)
                cont_thoughts[-1].append(sot)
                # for the first round, the "previous" hidden thoughts are just the
                # embeddings of all trivial detection events
                if i == 0:
                    # (B, N_L)
                    all_zeros = torch.zeros(
                        (src.shape[0], self.N_L), device=self.device, dtype=torch.int
                    )
                    # (B, N_L, d_m)
                    prev_embedding = self.error_embedding(all_zeros)
                    prev_embedding = self.error_positional_embedding(prev_embedding)
                else:
                    # (B, c, d_m)
                    prev_embedding = cont_thoughts[-2][-1]
                # generate c continuous thoughts
                for j in range(c):
                    if j == 0:
                        # input is just the <sot> embedding
                        input = cont_thoughts[-1][0]
                    else:
                        # input is <sot> embedding, plus everything else that's been
                        # generated so far
                        input = torch.cat(
                            (cont_thoughts[-1][0], cont_thoughts[-1][-1]), dim=1
                        )
                    cont_thoughts[-1].append(
                        self.decoder(
                            input,
                            prev_embedding,
                            encoder_output,
                        )
                    )
            else:
                # (B, N_L)
                logical_errors = tgt[
                    :, i * (self.N_L + 1) : i * (self.N_L + 1) + self.N_L
                ]
                assert logical_errors[1, 0] == 2
                if logical_errors.shape[1] > 1:
                    assert logical_errors[1, 1] != 2
                # (B, N_L, d_m)
                logical_errors_embedding = self.error_embedding(logical_errors)
                logical_errors_embedding = self.error_positional_embedding(
                    logical_errors_embedding
                )
                if num_ct_rounds > 0 and i == num_ct_rounds:
                    prev_logical_err_embedding = cont_thoughts[-1][-1]
                else:
                    if i == 0:
                        prev_logical_errors = torch.zeros_like(logical_errors)
                    else:
                        prev_logical_errors = tgt[
                            :,
                            (i - 1) * (self.N_L + 1) + 1 : (i - 1) * (self.N_L + 1)
                            + 1
                            + self.N_L,
                        ]
                        assert prev_logical_errors[1, 0] != 2
                    prev_logical_err_embedding = self.error_embedding(
                        prev_logical_errors
                    )
                    prev_logical_err_embedding = self.error_positional_embedding(
                        prev_logical_err_embedding
                    )
                decoder_output = self.decoder(
                    logical_errors_embedding,
                    prev_logical_err_embedding,
                    encoder_output,
                )
                # (B, N_L, 1)
                decoder_output = self.linear(decoder_output)
                decoder_output = self.dropout(decoder_output)
                # (B, N_L)
                decoder_output = decoder_output.squeeze()
                if len(decoder_output.shape) == 1:
                    decoder_output = decoder_output.unsqueeze(1)
                i -= num_ct_rounds
                all_decoder_outputs[:, i * self.N_L : (i + 1) * self.N_L] = (
                    decoder_output
                )
        return all_decoder_outputs

    def generate_predictions(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        num_ct_rounds: int,
        c: int,
    ) -> torch.Tensor:
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        num_rounds = src.shape[1] // self.N_D
        encoder_output = None
        cont_thoughts = []
        preds = (
            torch.ones(
                (src.shape[0], num_rounds - num_ct_rounds, self.N_L + 1),
                device=self.device,
                dtype=torch.int,
            )
            * 2
        )

        for i in range(num_rounds):
            if src_mask is not None:
                tmp_src_mask = src_mask[
                    i * self.N_D : (i + 1) * self.N_D, i * self.N_D : (i + 1) * self.N_D
                ]
            else:
                tmp_src_mask = None
            dets = src[:, i * self.N_D : (i + 1) * self.N_D]
            # (B, N_D, d_m)
            if i == 0:
                all_zeros = torch.zeros_like(dets)
                encoder_output = self.detector_embedding(all_zeros)
                encoder_output = self.detector_positional_embedding(encoder_output)
            embedded_dets = self.detector_embedding(dets)
            embedded_dets = self.detector_positional_embedding(embedded_dets)
            embedded_dets += encoder_output
            encoder_output = self.encoder(embedded_dets, tmp_src_mask)
            if i < num_ct_rounds:
                cont_thoughts.append([])
                # "start of thought" token
                sot = (
                    torch.ones((src.shape[0], 1), device=self.device, dtype=torch.int)
                    * 3
                )
                sot = self.error_embedding(sot)
                cont_thoughts[-1].append(sot)
                # for the first round, the "previous" hidden thoughts are just the
                # embeddings of all trivial detection events
                if i == 0:
                    all_zeros = torch.zeros(
                        (src.shape[0], self.N_L), device=self.device, dtype=torch.int
                    )
                    prev_embedding = self.error_embedding(all_zeros)
                    prev_embedding = self.error_positional_embedding(prev_embedding)
                else:
                    prev_embedding = cont_thoughts[-2][-1]
                # generate c continuous thoughts
                for j in range(c):
                    if j == 0:
                        # input is just the <sot> embedding
                        input = cont_thoughts[-1][0]
                    else:
                        # input is <sot> embedding, plus everything else that's been
                        # generated so far
                        input = torch.cat(
                            (cont_thoughts[-1][0], cont_thoughts[-1][-1]), dim=1
                        )
                    cont_thoughts[-1].append(
                        self.decoder(
                            input,
                            prev_embedding,
                            encoder_output,
                        )
                    )
            else:
                if num_ct_rounds > 0 and i == num_ct_rounds:
                    prev_logical_err_embedding = cont_thoughts[-1][-1]
                else:
                    if i == 0:
                        prev_logical_errors = torch.zeros_like(preds[:, 0, 1:])
                    else:
                        prev_logical_errors = preds[:, i - num_ct_rounds - 1, 1:]
                    prev_logical_err_embedding = self.error_embedding(
                        prev_logical_errors
                    )
                    prev_logical_err_embedding = self.error_positional_embedding(
                        prev_logical_err_embedding
                    )
                for j in range(self.N_L):
                    tgt_data = preds[:, i - num_ct_rounds, : j + 1]
                    tgt_embedding = self.error_embedding(tgt_data)
                    tgt_embedding = self.error_positional_embedding(tgt_embedding)
                    decoder_outputs = self.decoder(
                        tgt_embedding,
                        prev_logical_err_embedding,
                        encoder_output,
                    )
                    decoder_outputs = self.linear(decoder_outputs)
                    decoder_outputs = F.sigmoid(decoder_outputs)
                    decoder_outputs = (decoder_outputs > 0.5).int()
                    decoder_outputs = decoder_outputs.squeeze(-1)
                    preds[:, i - num_ct_rounds, j + 1] = decoder_outputs[:, -1]
        return preds[:, :, 1:]
