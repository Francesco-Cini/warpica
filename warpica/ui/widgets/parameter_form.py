"""A form with one number field for each parameter of a metric."""

from collections.abc import Sequence

from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QWidget

from warpica.ui.core.registry import Parameter


class ParameterForm(QWidget):
    """Number fields created from a list of ``Parameter`` descriptions."""

    def __init__(self) -> None:
        super().__init__()
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        # One field per parameter, kept by name so the values can be read back.
        self._fields: dict[str, QDoubleSpinBox] = {}

    def set_parameters(self, parameters: Sequence[Parameter]) -> None:
        """Replace the fields with new ones for ``parameters``.

        Parameters
        ----------
        parameters : sequence of Parameter
            The numbers to ask for, in the order they should appear.
        """
        # removeRow deletes both the label and the field of each row.
        while self._form.rowCount():
            self._form.removeRow(0)
        self._fields = {}

        if not parameters:
            self._form.addRow(QLabel("This metric has no parameters."))
            return

        for parameter in parameters:
            field = QDoubleSpinBox()
            field.setDecimals(3)
            field.setRange(parameter.minimum, parameter.maximum)
            field.setSingleStep(parameter.step)
            field.setValue(parameter.default)
            self._form.addRow(parameter.label, field)
            self._fields[parameter.name] = field

    def values(self) -> dict[str, float]:
        """Return the current value of every field, keyed by parameter name."""
        return {name: field.value() for name, field in self._fields.items()}
