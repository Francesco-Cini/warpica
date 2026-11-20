"""The main window: a sidebar of stages and the page for the selected one."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
)

from warpica.ui.core.state import Session, Stage
from warpica.ui.pages.metric_page import MetricPage
from warpica.ui.pages.results_page import ResultsPage
from warpica.ui.pages.solver_page import SolverPage

# Flags that let the user click an item. Removing them greys the item out.
_CLICKABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class MainWindow(QMainWindow):
    """Sidebar with one entry per stage, and the page of the chosen stage.

    A stage is greyed out in the sidebar until it has what it needs: the
    Solver needs a metric and the Results need a solved tensor. The state that
    decides this lives in ``self.session``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("warpica")

        self.session = Session()

        # The sidebar rows and the pages are both in the order of ``Stage``,
        # so row n always shows page n.
        self._sidebar = QListWidget()
        self._sidebar.setMinimumWidth(140)
        self._sidebar.setMaximumWidth(400)
        self._pages = QStackedWidget()
        self._solver_page = SolverPage(self.session)
        self._results_page = ResultsPage(self.session)
        for page in (
            MetricPage(self.session),
            self._solver_page,
            self._results_page,
        ):
            self._sidebar.addItem(QListWidgetItem())
            self._pages.addWidget(page)

        self._sidebar.currentRowChanged.connect(self._pages.setCurrentIndex)
        self.session.subscribe(self._refresh_sidebar)
        self._refresh_sidebar()
        self._sidebar.setCurrentRow(0)

        # The splitter lets the user drag the edge of the sidebar.
        splitter = QSplitter()
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._pages)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop running calculations, so no process is left working after we close."""
        self._solver_page.cancel()
        self._results_page.cancel()
        super().closeEvent(event)

    def _refresh_sidebar(self) -> None:
        """Update every sidebar entry to match the session."""
        stages = list(Stage)
        for row, stage in enumerate(stages):
            item = self._sidebar.item(row)
            item.setText(f"{stage.value}\n{self.session.status(stage)}")
            if self.session.is_unlocked(stage):
                item.setFlags(item.flags() | _CLICKABLE)
            else:
                item.setFlags(item.flags() & ~_CLICKABLE)

        # If the page being shown just became locked (for example the metric
        # was rebuilt while the Results were open), go back to the Metric page.
        current = self._sidebar.currentRow()
        if current >= 0 and not self.session.is_unlocked(stages[current]):
            self._sidebar.setCurrentRow(0)
