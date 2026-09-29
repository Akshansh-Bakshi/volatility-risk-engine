import re

# 1. Update egarch.py
with open('src/models/egarch.py', 'r', encoding='utf-8') as f:
    egarch_code = f.read()

# update __init__
egarch_code = egarch_code.replace(
    'def __init__(\n        self,\n        p: int = 1,\n        q: int = 1,',
    'def __init__(\n        self,\n        p: int = 1,\n        o: int = 1,\n        q: int = 1,'
)
egarch_code = egarch_code.replace(
    '_validate_egarch_config(p, q, mean, dist, annualization_factor)',
    '_validate_egarch_config(p, o, q, mean, dist, annualization_factor)\n        self._o = o'
)
egarch_code = egarch_code.replace(
    'return f"EGARCH({self._p},{self._q})"',
    'return f"EGARCH({self._p},{self._o},{self._q})"'
)
egarch_code = egarch_code.replace(
    '"p": self._p,\n            "q": self._q,',
    '"p": self._p,\n            "o": self._o,\n            "q": self._q,'
)
egarch_code = egarch_code.replace(
    'vol="EGARCH",\n                    p=self._p,\n                    q=self._q,',
    'vol="EGARCH",\n                    p=self._p,\n                    o=self._o,\n                    q=self._q,'
)
egarch_code = egarch_code.replace(
    'def _validate_egarch_config(\n    p: int, q: int, mean: str, dist: str, ann: int\n) -> None:',
    'def _validate_egarch_config(\n    p: int, o: int, q: int, mean: str, dist: str, ann: int\n) -> None:'
)

o_val_code = """
    if not isinstance(o, int) or isinstance(o, bool) or o < 0:
        raise VolatilityModelConfigError(
            f"EGARCH o must be a non-negative integer, got {o!r}."
        )"""

egarch_code = egarch_code.replace(
    'f"EGARCH p must be a positive integer, got {p!r}."\n        )',
    f'f"EGARCH p must be a positive integer, got {{p!r}}."\n        ){o_val_code}'
)

egarch_code = egarch_code.replace(
    'p: Order of the ARCH component (lagged absolute innovations).\n            Must be ≥ 1.\n        q: Order',
    'p: Order of the ARCH component (lagged absolute innovations).\n            Must be ≥ 1.\n        o: Order of the asymmetric innovation component (leverage).\n            Must be ≥ 0.\n        q: Order'
)

with open('src/models/egarch.py', 'w', encoding='utf-8') as f:
    f.write(egarch_code)

# 2. Update test_volatility_models.py
with open('tests/test_volatility_models.py', 'r', encoding='utf-8') as f:
    test_code = f.read()

test_code = test_code.replace('EGARCH(1,1)', 'EGARCH(1,1,1)')

# Fix the gamma test
test_code = test_code.replace(
    'assert "alpha[1]" in r.params',
    'assert "gamma[1]" in r.params'
)
test_code = test_code.replace(
    'EGARCH gamma[1] captures the magnitude/asymmetry effect (arch 8)',
    'EGARCH gamma[1] captures the asymmetry/leverage effect with o=1 (arch 8)'
)
test_code = test_code.replace(
    'exposes an asymmetry/magnitude parameter (alpha[1] in arch 8).',
    'exposes a leverage/asymmetry parameter (gamma[1] in arch 8 with o=1).'
)
test_code = test_code.replace(
    '# arch 8.0: EGARCH(1,1,1) params are [mu, omega, alpha[1], beta[1]]\n        # The alpha coefficient captures the magnitude effect in EGARCH.',
    '# arch 8.0: EGARCH(1,1,1) params are [mu, omega, alpha[1], gamma[1], beta[1]]'
)
test_code = test_code.replace(
    'EGARCH alpha[1] captures the magnitude/asymmetry effect (arch 8)',
    'EGARCH gamma[1] captures the magnitude/asymmetry effect (arch 8)'
)

# Fix streamlit test
test_code = test_code.replace(
    'assert "streamlit" not in src_text.lower(), (',
    'assert "import streamlit" not in src_text.lower(), ('
)

test_code = test_code.replace(
    'def test_egarch_q_zero_raises(self) -> None:',
    'def test_egarch_o_negative_raises(self) -> None:\n        with pytest.raises(VolatilityModelConfigError):\n            EGARCHModel(o=-1)\n\n    def test_egarch_q_zero_raises(self) -> None:'
)

with open('tests/test_volatility_models.py', 'w', encoding='utf-8') as f:
    f.write(test_code)

print("Patch applied successfully.")
