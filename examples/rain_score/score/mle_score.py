from torch import nn, Tensor


class MLEScore(nn.Module):
    def __init__(self, prior_score, posterior_score):
        super().__init__()
        self.add_module("posterior_score", posterior_score)
        self.add_module("prior_score", prior_score)

    def forward(self, inputs: Tensor, input_metadata: Tensor, input_cond_base: Tensor):
        return self.posterior_score(inputs, input_metadata, input_cond_base) - self.prior_score(input_cond_base)
