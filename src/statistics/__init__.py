"""Statistical diagnostics of return series.

Responsibility: descriptive statistics and hypothesis tests that characterise
returns (distribution shape, autocorrelation, volatility clustering) before any
model is fitted.

Note: this subpackage shares its name with the standard-library ``statistics``
module.  That is harmless because all imports in this project are absolute,
but ``src/`` itself must never be placed on ``sys.path``.

Stage 4 — EDA / Dataset Profiling
----------------------------------
Public API::

    from src.statistics.profile import DatasetProfile, build_dataset_profile
    from src.statistics.descriptive import ReturnStats, compute_return_stats
    from src.statistics.figures import (
        plot_price_series,
        plot_return_series,
        plot_return_histogram,
        plot_data_quality,
    )
    from src.statistics.snapshot import save_eda_snapshot

Stage 5 — Statistical Diagnostics
-----------------------------------
Public API::

    from src.statistics.diagnostics import (
        TestResult,
        ReturnDiagnostics,
        run_adf_test,
        run_kpss_test,
        run_ljung_box_returns,
        run_ljung_box_squared,
        run_arch_lm_test,
        run_all_diagnostics,
        plot_acf,
        plot_pacf,
    )
"""
