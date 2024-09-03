"""
Changing T5Attention's forward to support prefix tuning, along with subclassing other classes that
use T5Attention. Changes in T5Attention's forward from are marked with
"# <CHANGE>" and "# </CHANGE>". It's possible that the added logic can be separated as some code
that entirely preceeds the original forward, s.t. we can call super().forward() without code
duplciation. Even better, we might be able to use a pre-hook so that most of this won't be needed.
"""


# https://github.com/allenai/better-promptability/blob/main/better_promptability/models/t5_with_prefix.py#L260
# https://github.com/huggingface/transformers/issues/15591


import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5Stack,
    T5ForConditionalGeneration,
)

import matplotlib.pyplot as plt


from typing import List, Optional, Tuple, Union


class T5WithPrefixConfig(T5Config):
    def __init__(
        self, num_prefix=None, reparam=False, reparam_dim=512, no_decoder_self_attn=False,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_prefix = num_prefix
        self.reparam = reparam
        self.reparam_dim = reparam_dim
        self.no_decoder_self_attn = no_decoder_self_attn

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        config_dict, kwargs = T5Config.get_config_dict(*args, **kwargs)
        for field in ("num_prefix", "reparam_dim"):
            assert field not in config_dict
            if field in kwargs:
                config_dict[field] = kwargs.pop(field)
        return config_dict, kwargs


class T5AttentionWithPrefix(T5Attention):
    def __init__(self, config: T5WithPrefixConfig, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.num_prefix = config.num_prefix

        self.config = config

    # fmt: off

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).  # noqa: E501
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"  # noqa: E501
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states


        ################################################
        # shared prompt: 1 x (num_heads x num_prefix x d_kv)

        sample = 16
        n_channels = self.n_channels
        _, seq_length, d_model = hidden_states.shape
        batch_size_real = batch_size // n_channels

        # TODO: ignore MPT when doing deep prompt?
        if self.config.multivariate_projection == 'attention':
            # TODO: can average across channel dimension in the end?
            # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over

            # the plan:
            # original input: (batch*channel) x time x d_model
            # make it batch x channel x time x d_model
            # sample to 16 time steps
                # TODO: change sample to smoothing?
                # TODO: ignore possible padding
            # transpose to (batch*time) x channel x d_model
            # do multihead attention
            # average across channels, and reshape, so output is batch x (time*d_model)
            # linear, unflatten to batch x 1 x num_heads x num_prefix x d_kv
            # repeat and unflatten, so output is (batch*channel) x n_heads x num_prefix x d_kv

            # key
            # TODO: is reshape invertible?

            # hidden_states: bs*channel x seq_length x d_model

            hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
            # hidden_states_proj = hidden_states_[:, :, ::seq_length//sample, :].transpose(1, 2).reshape(-1, n_channels, d_model)
            hidden_states_proj = hidden_states_.transpose(1, 2).reshape(-1, n_channels, d_model)
            attn_output, attn_output_weights = self.shared_prompt_projection['mha'](self.shared_prompt_projection['q'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['k'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['v'](hidden_states_proj))
            # reshape, make it channel x d_kv x time
            attn_output = attn_output.reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)
            shared_prompt_projection_k = self.shared_prompt_projection['linear_key'](attn_output) # bs x channel x d_kv x (num_prefix*n_heads)
            # shared_prompt_projection = shared_prompt_projection.permu
            shared_prompt_projection_key = shared_prompt_projection_k.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.num_heads).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.num_heads, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)
            #             # bsz, num_heads, num_prefix, d_kv


            shared_prompt_projection_v = self.shared_prompt_projection['linear_value'](attn_output)
            shared_prompt_projection_value = shared_prompt_projection_v.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.num_heads).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.num_heads, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)

            if self.config.visualize_attention:
                scores = attn_output_weights.mean(dim=0)
                plt.imshow(scores.detach().cpu().numpy())
                plt.savefig(f"plots/attention/{self.layer_number}.png")


        elif self.config.multivariate_projection == 'linear':
            hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
            hidden_states_proj = hidden_states_[:, :, ::seq_length//sample, :].reshape(-1, n_channels, sample*d_model)
            shared_prompt_projection_key = self.shared_prompt_projection['unflatten'](torch.mean(self.shared_prompt_projection['linear'](hidden_states_proj), dim=1)).repeat(1, n_channels, 1, 1, 1).flatten(0, 1)
            shared_prompt_projection_value = self.shared_prompt_projection['unflatten'](torch.mean(self.shared_prompt_projection['linear'](hidden_states_proj), dim=1)).repeat(1, n_channels, 1, 1, 1).flatten(0, 1)

        else:
            raise ValueError('Invalid projection type')
        ################################################


        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # <CHANGE>
        # move this up to not include layer-specific prefix
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None

        # prefix tuning

        # if self.config.prefix_tuning:
        #     key_states = torch.cat([self.prefix_key, shared_prompt_projection_key, key_states], dim=2)     # self.prefix_key: bsz, num_heads, num_prefix, d_kv
        #     value_states = torch.cat([self.prefix_value, shared_prompt_projection_value, value_states], dim=2)
        # else:
        key_states = torch.cat([shared_prompt_projection_key, key_states], dim=2)     # self.prefix_key: bsz, num_heads, num_prefix, d_kv
        value_states = torch.cat([shared_prompt_projection_value, value_states], dim=2)



        # key_states = torch.cat([self.prefix_key, key_states], dim=2)     # self.prefix_key: bsz, num_heads, num_prefix, d_kv
        # value_states = torch.cat([self.prefix_value, value_states], dim=2)


        if mask is not None:
            prefix_mask = torch.zeros(
                # batch_size, 1, mask.size(2), self.num_prefix*(2 if self.config.prefix_tuning else 1), device=hidden_states.device
                batch_size, 1, mask.size(2), self.num_prefix*(n_channels), device=hidden_states.device
                # batch_size, 1, mask.size(2), self.num_prefix, device=hidden_states.device
            )
            mask = torch.cat([prefix_mask, mask], dim=-1)
        # </CHANGE>

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9


        # TODO: since there is no order, shouldn't be here?
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            # <CHANGE>
            position_bias = torch.cat(
                [
                    torch.zeros(
                        # position_bias.shape[:3] + (self.num_prefix*(2 if self.config.prefix_tuning else 1),),
                        position_bias.shape[:3] + (self.num_prefix*(n_channels),),
                        # position_bias.shape[:3] + (self.num_prefix,),
                        device=position_bias.device,
                    ),
                    position_bias,
                ],
                dim=3,
            )
            # </CHANGE>

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias

        # position_bias = None

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)


        # <CHANGE>  moved one line up
        # </CHANGE>
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    # fmt: on


class T5LayerSelfAttentionWithPrefix(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        if not config.no_decoder_self_attn:
            self.SelfAttention = T5AttentionWithPrefix(
                config, has_relative_attention_bias=has_relative_attention_bias
            )


class T5LayerCrossAttentionWithPrefix(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = T5AttentionWithPrefix(config, has_relative_attention_bias=False)


class T5BlockWithPrefix(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, layer_number=None):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer[0] = T5LayerSelfAttentionWithPrefix(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[0].SelfAttention.layer_number = layer_number
        if self.is_decoder:
            self.layer[1] = T5LayerCrossAttentionWithPrefix(config)
            # TODO: use prompt for decoder?
            # self.layer[1] = T5LayerCrossAttention(config)
            self.layer[1].EncDecAttention.layer_number = layer_number


class T5StackWithPrefixMulti(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        # prefix tuning - reparam 'trick'
        self.input_tokens = torch.arange(self.config.num_prefix)
        per_layer_dim = self.config.num_heads * self.config.d_kv
        total_dim = self.config.num_layers * 2 * per_layer_dim


        # if self.config.prefix_tuning:
        #     self.prefix_embed = (
        #         nn.Sequential(
        #             nn.Embedding(self.config.num_prefix, per_layer_dim),
        #             nn.Linear(per_layer_dim, self.config.reparam_dim),
        #             nn.Tanh(),
        #             nn.Linear(self.config.reparam_dim, total_dim),
        #         )
        #         if self.config.reparam
        #         else nn.Embedding(self.config.num_prefix, total_dim)
        #     )
        #     if self.is_decoder:
        #         self.prefix_embed_cross = (
        #             nn.Sequential(
        #                 nn.Embedding(self.config.num_prefix, per_layer_dim),
        #                 nn.Linear(per_layer_dim, self.config.reparam_dim),
        #                 nn.Tanh(),
        #                 nn.Linear(self.config.reparam_dim, total_dim),
        #             )
        #             if self.config.reparam
        #             else nn.Embedding(self.config.num_prefix, total_dim)
        #         )


        self.block = torch.nn.ModuleList(
            [
                T5BlockWithPrefix(self.config, has_relative_attention_bias=bool(i == 0), layer_number=i)
                for i in range(self.config.num_layers)
            ]
        )

        # T5Stack has a self.init_weights() call here, but it's repetitive since we do it in
        # T5ForConditionalGenerationWithPrefix anyway.


        # shared prompt: 1 x (num_heads x num_prefix x d_kv)
        # hidden_states: bsz x seq_len x dim: 7*64*64
        # just learn 1D output, reshape to 1 x (num_heads x num_prefix x d_kv)

        if config.multivariate_projection == 'linear':
            # self.shared_prompt_projection = nn.Sequential(
            #                                         nn.Flatten(start_dim=0),
            #                                         nn.Linear(7 * \
            #                                             (config.num_patches + (config.num_prefix if config.MPT else 0)) * \
            #                                                 config.d_model, config.reparam_dim),
            #                                         nn.Tanh(),
            #                                         nn.Linear(config.reparam_dim, config.num_heads * config.num_prefix * config.d_kv),
            #                                         nn.Unflatten(0, (1, config.num_heads, config.num_prefix, config.d_kv))
            #                                     )

            # the plan:
            # original input: channel x time x d_model
            # sample to 16 time steps
            #           -- so no dependence on sequence length
            #   so channel x 16 x d_model
            #  flatten: channel x 16*d_model


            # transpose to time x channel x d_model
            # do multihead attention
            # average across channels, so output is 1 x (time x 1 x d_model)
            # flatten, linear, unflatten

            dim, sample = config.d_model, 16

            self.shared_prompt_projection_linear = nn.Linear(sample*dim, config.num_heads * config.num_prefix * config.d_kv)
            self.unflatten = nn.Unflatten(1, (1, config.num_heads, config.num_prefix, config.d_kv))

            self.shared_prompt_projection = {
                'linear': self.shared_prompt_projection_linear,
                'unflatten': self.unflatten
            }




        # elif config.multivariate_projection == 'conv':
        #     # input: torch.Size([7, 80, 1024])

        #     # TODO: not variable number of channels

        #     # sample to 16 time steps, or no sample
        #     # do 1d conv
        #     # channel: time, length: variable
        #     self.shared_prompt_projection = nn.Sequential(
        #         Sample(),
        #         nn.Linear(config.d_model, 1),
        #         nn.Flatten(start_dim=1),
        #         Transpose(),
        #         nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3),

        #     )


        elif config.multivariate_projection == 'attention':
            # TODO: can average across channel dimension in the end?
            # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over


            # the plan:
            # original input: channel x time x d_model
            # sample to 16 time steps
            #           -- so no dependence on sequence length
            # transpose to time x channel x d_model
            # do multihead attention
            # average across channels, so output is 1 x (time x 1 x d_model)
            # flatten, linear, unflatten
            # dim, sample = config.d_model, 16
            # TODO: instead of sampling, do projection to same space (like itransformer) But this makes it necessary to have same sequence length? should be fine because you split and pad anyways
            # itransformer: embed the whole series into one vector, each channel independently. Then do attention, using this vector as feature
            # So, you should take the original time series (not the encoding), sample it?, then do attention
            # what I have here is to encode at each timestamp
            self.shared_prompt_projection_k = nn.Linear(config.d_model, config.d_kv, bias=False)
            self.shared_prompt_projection_q = nn.Linear(config.d_model, config.d_kv, bias=False)  # TODO: should dim be smaller than d_kv?
            self.shared_prompt_projection_v = nn.Linear(config.d_model, config.d_kv, bias=False)

            # batch size: time, length: variable, E: 1024
            self.shared_prompt_projection_mha = nn.MultiheadAttention(config.d_kv, num_heads=1, batch_first=True,
                                                                      kdim=config.d_kv, vdim=config.d_kv)
            # out: time x channel x d_kv

            # reshape, make it bs x channel x d_kv x time

            # self.shared_prompt_projection_linear = nn.Linear(sample*dim, config.num_heads * config.num_prefix * config.d_kv)
            # https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
            
            
            size = config.forecast_length+1 if config.is_decoder else config.seq_len
            
            self.shared_prompt_projection_linear_key = nn.Linear(size, config.num_prefix*config.num_heads)  # TODO: move this to before attention?
            self.shared_prompt_projection_linear_value = nn.Linear(size, config.num_prefix*config.num_heads)
            # out: bs x channel x d_kv x (num_prefix*n_heads)

            # reshape, make it bs x 1 x 1 x (channel * config.num_prefix) x config.d_kv
            #


            # self.unflatten = nn.Unflatten(1, (1, config.num_heads, config.num_prefix, config.d_kv))
            # self.unflatten = nn.Unflatten(1, (1, config.num_prefix, config.d_kv))

            self.shared_prompt_projection = {
                'k': self.shared_prompt_projection_k,
                'q': self.shared_prompt_projection_q,
                'v': self.shared_prompt_projection_v,
                'mha': self.shared_prompt_projection_mha,
                'linear_key': self.shared_prompt_projection_linear_key,
                'linear_value': self.shared_prompt_projection_linear_value,
                # 'unflatten': self.unflatten
            }



            # self.shared_prompt_projection = nn.Sequential(
            #                                         Sample(),
            #                                         # nn.Linear(config.d_model, 1),
            #                                         # nn.Flatten(start_dim=1),
            #                                         nn.MultiheadAttention(16, num_heads=1, batch_first=False),  # batch size: time, length: variable, E: 1024
            #                                         # TODO: tanh? relu?

            #                                         nn.Linear(16*7, 1),
            #                                         # TODO: sequence length is not even variable
            # )

        else:
            raise ValueError('Invalid projection type')


    class Transpose(torch.nn.Module):
        def forward(self, x):
            return x.T

    class Sample(torch.nn.Module):
        def forward(self, x):
            return x[:, ::(len(x)//16)]


    def generate_prefix_item(self, input_ids, embedding):
        bsz = input_ids.size(0)
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(input_ids.device)
        prefix = embedding(input_tokens)  # batch, seq, layer * embed * 2
        prefix = prefix.view(
            bsz,
            self.config.num_prefix,
            self.config.num_layers,
            2,
            self.config.num_heads,
            self.config.d_kv,
        )
        prefix = prefix.permute([3, 2, 0, 4, 1, 5])  # 2, num_layers, bsz, num_heads, num_prefix, d_kv
        return prefix[0], prefix[1]

    ################################################
    # def forward(self, input_ids=None, **kwargs):
        # prefix_key, prefix_value = self.generate_prefix_item(input_ids, self.prefix_embed)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # is this a reference or copy?
        # should be a reference. weigths are updated. Also id(self.shared_prompt_projection) are the same
        # why do both self and cross attention?
        # self attention is for encoder, cross is for decoder (not used)
        for block in self.block:
            for i, layer in enumerate(block.layer):
                if isinstance(layer, T5LayerSelfAttentionWithPrefix):
                    layer.SelfAttention.shared_prompt_projection = self.shared_prompt_projection
                    layer.SelfAttention.n_channels = self.config.n_channels

                # TODO: need prompt for decoder?
                if isinstance(layer, T5LayerCrossAttentionWithPrefix):
                    layer.EncDecAttention.shared_prompt_projection = self.shared_prompt_projection
                    layer.EncDecAttention.n_channels = self.config.n_channels

        # if self.config.prefix_tuning:

        #     prefix_key, prefix_value = self.generate_prefix_item(input_ids, self.prefix_embed)
        #     kwargs['use_cache'] = False
        # ################################################
        #     prefix_key_cross = prefix_value_cross = [None] * len(prefix_key)
        #     if self.is_decoder:
        #         prefix_key_cross, prefix_value_cross = self.generate_prefix_item(
        #             input_ids, self.prefix_embed_cross
        #         )
        #     for block, k, v, k_cross, v_cross in zip(
        #         self.block, prefix_key, prefix_value, prefix_key_cross, prefix_value_cross
        #     ):
        #         for i, layer in enumerate(block.layer):
        #             if isinstance(layer, T5LayerSelfAttentionWithPrefix):
        #                 layer.SelfAttention.prefix_key = k
        #                 layer.SelfAttention.prefix_value = v
        #             if isinstance(layer, T5LayerCrossAttentionWithPrefix):
        #                 layer.EncDecAttention.prefix_key = k_cross
        #                 layer.EncDecAttention.prefix_value = v_cross


        output = super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        self.clean_up()
        return output



    def clean_up(self):
        pass
        # For safety, in case other code uses it
        # if self.config.prefix_tuning:
        #     for block in self.block:
        #         for layer in block.layer:
        #             if isinstance(layer, T5LayerSelfAttentionWithPrefix):
        #                 del layer.SelfAttention.prefix_key
        #                 del layer.SelfAttention.prefix_value
        #             if isinstance(layer, T5LayerCrossAttentionWithPrefix):
        #                 del layer.EncDecAttention.prefix_key
        #                 del layer.EncDecAttention.prefix_value


class T5ForConditionalGenerationWithPrefixMulti(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = T5StackWithPrefixMulti(self.encoder.config, self.shared)
        self.decoder = T5StackWithPrefixMulti(self.decoder.config, self.shared)
        self.init_weights()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        
        if input_ids is not None and input_ids.dim() == 3:
            input_ids = input_ids.reshape(
                (-1, input_ids.shape[2])
            )
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.reshape(
                (-1, attention_mask.shape[2])
            )
        if labels is not None and labels.dim() == 3:
            labels = labels.reshape(
                (-1, labels.shape[2])
            )

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
