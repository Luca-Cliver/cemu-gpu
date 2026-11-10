import torch
import torch.nn.functional as functional

from flexgen.pytorch_backend import DeviceType, TorchDevice, TorchTensor


class OptTorchDevice(TorchDevice):
    def __init__(self, name, *, pin_memory=True, layer_norm_eps=1e-5):
        super().__init__(name)
        self.pin_host_memory = pin_memory
        self.layer_norm_eps = layer_norm_eps
        self.output_observer = None

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU and not self.pin_host_memory:
            pin_memory = False
        return super().allocate(shape, dtype, pin_memory, name)

    def opt_project(self, inputs, weights, heads):
        query_weight, query_bias, key_weight, key_bias, value_weight, value_bias = weights[:6]
        norm_weight, norm_bias = weights[8:10]
        batch, tokens, hidden = inputs.shape
        dimension = hidden // heads
        normalized = functional.layer_norm(inputs.data, (hidden,), norm_weight.data,
                                          norm_bias.data, self.layer_norm_eps)
        query = functional.linear(normalized, query_weight.data, query_bias.data) * dimension ** -0.5
        keys = functional.linear(normalized, key_weight.data, key_bias.data)
        values = functional.linear(normalized, value_weight.data, value_bias.data)
        query = query.view(batch, tokens, heads, dimension).permute(0, 2, 1, 3)
        keys = keys.view(batch, tokens, heads, dimension).permute(1, 0, 2, 3)
        values = values.view(batch, tokens, heads, dimension).permute(1, 0, 2, 3)
        return query, keys, values

    def opt_finish(self, inputs, output, output_weight, output_bias, donate=True):
        batch, tokens, hidden = inputs.shape
        output = output.transpose(1, 2).reshape(batch, tokens, hidden)
        result = functional.linear(output, output_weight.data, output_bias.data) + inputs.data
        if donate:
            inputs.delete()
        return TorchTensor.create_from_torch(result, self)

    def mha(self, inputs, mask, query_weight, query_bias, key_weight, key_bias,
            value_weight, value_bias, output_weight, output_bias, norm_weight,
            norm_bias, heads, donate, compress_cache, comp_config):
        if compress_cache or query_weight.device.device_type == DeviceType.COMPRESSED:
            raise NotImplementedError("OPT adapter currently requires uncompressed weights and KV")
        weights = (query_weight, query_bias, key_weight, key_bias, value_weight,
                   value_bias, output_weight, output_bias, norm_weight, norm_bias)
        query, keys, values = self.opt_project(inputs, weights, heads)
        batch, tokens, hidden = inputs.shape
        dimension = hidden // heads
        key_matrix = keys.permute(1, 2, 3, 0)
        scores = torch.matmul(query, key_matrix)
        causal = torch.ones((tokens, tokens), device=self.dev, dtype=torch.bool).tril()
        allowed = causal[None, None] & mask.data[:, None, None, :]
        scores = scores.masked_fill(~allowed, -1e4)
        probabilities = functional.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(probabilities, values.permute(1, 2, 0, 3))
        result = self.opt_finish(inputs, output, output_weight, output_bias, donate[0])
        if donate[1]:
            mask.delete()
        return (result,
                TorchTensor.create_from_torch(keys.reshape(tokens, batch * heads, dimension), self),
                TorchTensor.create_from_torch(values.reshape(tokens, batch * heads, dimension), self))

    def mha_gen(self, inputs, mask, query_weight, query_bias, key_weight, key_bias,
                value_weight, value_bias, output_weight, output_bias, norm_weight,
                norm_bias, heads, key_cache, value_cache, donate, attn_sparsity,
                compress_cache, comp_config):
        if compress_cache or attn_sparsity != 1 or not isinstance(key_cache, TorchTensor):
            raise NotImplementedError("OPT reference Decode requires dense, uncompressed tensor caches")
        weights = (query_weight, query_bias, key_weight, key_bias, value_weight,
                   value_bias, output_weight, output_bias, norm_weight, norm_bias)
        query, keys, values = self.opt_project(inputs, weights, heads)
        batch, _, hidden = inputs.shape
        dimension = hidden // heads
        tokens = mask.shape[1]
        new_keys = keys.reshape(1, batch * heads, dimension)
        new_values = values.reshape(1, batch * heads, dimension)
        key_cache.data[tokens - 1:tokens].copy_(new_keys)
        value_cache.data[tokens - 1:tokens].copy_(new_values)
        history_keys = key_cache.data[:tokens].to(self.dev).view(tokens, batch, heads, dimension)
        history_values = value_cache.data[:tokens].to(self.dev).view(tokens, batch, heads, dimension)
        scores = torch.matmul(query, history_keys.permute(1, 2, 3, 0))
        scores = scores.masked_fill(~mask.data[:, None, None, :].to(self.dev), -1e4)
        probability = functional.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(probability, history_values.permute(1, 2, 0, 3))
        result = self.opt_finish(inputs, output, output_weight, output_bias, donate[0])
        if donate[1]:
            mask.delete()
        return (result, TorchTensor.create_from_torch(new_keys, self),
                TorchTensor.create_from_torch(new_values, self))

    def mlp(self, inputs, input_weight, input_bias, output_weight, output_bias,
            norm_weight, norm_bias, donate):
        normalized = functional.layer_norm(inputs.data, (inputs.shape[-1],),
                                          norm_weight.data, norm_bias.data, self.layer_norm_eps)
        activated = functional.relu(functional.linear(normalized, input_weight.data, input_bias.data))
        output = functional.linear(activated, output_weight.data, output_bias.data) + inputs.data
        if donate[0]:
            inputs.delete()
        return TorchTensor.create_from_torch(output, self)

    def opt_output_embed(self, inputs, norm_weight, norm_bias, token_weight, donate,
                         do_sample, temperature):
        normalized = functional.layer_norm(inputs.data, (inputs.shape[-1],),
                                          norm_weight.data, norm_bias.data, self.layer_norm_eps)
        logits = functional.linear(normalized[:, -1], token_weight.data)
        if self.output_observer is not None:
            self.output_observer(logits.detach())
        if do_sample and temperature >= 1e-5:
            tokens = torch.multinomial(functional.softmax(logits / temperature, dim=-1), 1)
        else:
            tokens = logits.argmax(dim=-1, keepdim=True)
        if donate[0]:
            inputs.delete()
        return TorchTensor.create_from_torch(tokens, self)
