import numpy as np
from scipy.stats import norm


class BlackScholesOption:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

    def _safe_sqrt(self, x):
        """Prevent sqrt(0) divide errors."""
        return np.sqrt(np.maximum(x, 1e-16))

    def d1(self):
        T_safe = np.maximum(self.T, 1e-16)
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * T_safe) / (
            self.sigma * self._safe_sqrt(T_safe)
        )

    def d2(self):
        T_safe = np.maximum(self.T, 1e-16)
        return self.d1() - self.sigma * self._safe_sqrt(T_safe)

    def price(self):
        """Return option price, handles T=0 as intrinsic value."""
        if np.isscalar(self.T) and self.T == 0:
            return max(
                0.0,
                (self.S - self.K) if self.option_type == "call" else (self.K - self.S),
            )
        else:
            d1 = self.d1()
            d2 = self.d2()
            if self.option_type == "call":
                return self.S * norm.cdf(d1) - self.K * np.exp(
                    -self.r * self.T
                ) * norm.cdf(d2)
            else:
                return self.K * np.exp(-self.r * self.T) * norm.cdf(
                    -d2
                ) - self.S * norm.cdf(-d1)

    def delta(self):
        """Return option delta, handles T=0 as step function."""
        if np.isscalar(self.T) and self.T == 0:
            return (
                1.0
                if (self.option_type == "call" and self.S > self.K)
                else -1.0
                if (self.option_type == "put" and self.S < self.K)
                else 0.0
            )
        else:
            d1 = self.d1()
            return norm.cdf(d1) if self.option_type == "call" else norm.cdf(d1) - 1


# Example usage
if __name__ == "__main__":
    opt1 = BlackScholesOption(
        S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call"
    )
    print("Call price:", opt1.price(), "Delta:", opt1.delta())

    opt2 = BlackScholesOption(
        S=100, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call"
    )
    print("Call price T=0:", opt2.price(), "Delta T=0:", opt2.delta())

    opt3 = BlackScholesOption(S=100, K=100, T=0.0, r=0.05, sigma=0.2, option_type="put")
    print("Put price T=0:", opt3.price(), "Delta T=0:", opt3.delta())
