"""Optional desktop interface for warpica.

This subpackage needs the ``ui`` extra (``pip install "warpica[ui]"``) and is
not part of the stable public API. ``import warpica`` never imports it.

Run it with ``python -m warpica.ui``.

The code is split in two on purpose:

``warpica.ui.core``
    Plain Python with no Qt imports. It describes the metrics, runs the
    calculation and prepares the numbers to plot. It can be tested without a
    screen.
``warpica.ui.windows`` and ``warpica.ui.pages``
    The Qt code: the main window with its sidebar, and one page for each
    stage (Metric, Solver, Results). They only collect input, call ``core``
    and draw the result.
"""
