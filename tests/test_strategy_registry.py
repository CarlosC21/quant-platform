from quant_platform.runner.strategy_factory import (
    register_strategy,
    create_strategy,
)


class DummyStrategy:
    def __init__(self, alpha=1, beta=2):
        self.alpha = alpha
        self.beta = beta

    def on_start(self, ctx):
        pass

    def on_bar(self, ctx, bar):
        pass

    def on_end(self, ctx):
        pass


def test_register_and_create_strategy_basic():
    register_strategy("dummy", DummyStrategy)
    s = create_strategy("dummy")
    assert isinstance(s, DummyStrategy)
    assert s.alpha == 1 and s.beta == 2


def test_register_with_params():
    register_strategy("dummy2", DummyStrategy)
    s = create_strategy("dummy2", {"alpha": 10, "beta": 20})
    assert s.alpha == 10
    assert s.beta == 20


def test_missing_strategy_raises():
    try:
        create_strategy("not_exists")
    except KeyError as e:
        assert "not registered" in str(e)
    else:
        assert False, "Expected KeyError"
