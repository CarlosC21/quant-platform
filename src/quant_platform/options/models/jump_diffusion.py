# src/quant_platform/options/models/jump_diffusion.py
class JumpDiffusionOption:
    def __init__(self, S, K, T, r, sigma, lambda_, muJ, sigmaJ, option_type="call"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lambda_ = lambda_
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.option_type = option_type

    def price(self):
        raise NotImplementedError("Merton jump-diffusion pricing to be implemented")
