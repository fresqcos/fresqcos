import numpy as np

class Receiver:

    def __init__(self,*, transmittance: float, x_basis_loss: Optional[float] = None, z_basis_loss: Optional[float] = None):

        self.transmittance = transmittance


        if x_basis_loss is None:
            self.x_basis_loss = 0

        else:
            self.x_basis_loss = x_basis_loss

        if z_basis_loss is None:
            self.z_basis_loss = self.x_basis_loss

        else:
            self.z_basis_loss = z_basis_loss

    @property
    def transmittance(self) -> float:
        """ Return the transmittance of the detector.

        Must be non-negative and less than 1
        """
        return self._transmittance

    @transmittance.setter
    def transmittance(self, value: float) -> None:
        if value < 0 or value >1:
            raise ValueError(f"transmittance must be non-negative and less than 1, got {value}")

        self._transmittance = float(value)

    @property
    def x_basis_loss(self) -> float:
        """ Return the optical loss in the X basis.

        Must be non-negative.
        """
        return self._x_basis_loss

    @x_basis_loss.setter
    def x_basis_loss(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"x_basis_loss must be non-negative, got {value}")

        self._x_basis_loss = float(value)

    @property
    def z_basis_loss(self) -> float:
        """ Return the optical loss in the Z basis.

        Must be non-negative.
        """
        return self._z_basis_loss

    @z_basis_loss.setter
    def z_basis_loss(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"z_basis_loss must be non-negative, got {value}")

        self._z_basis_loss = float(value)

    def x_basis_transmittance(self):

        return 10**(-self.x_basis_loss/10)*self.transmittance

    def z_basis_transmittance(self):

        return 10**(-self.z_basis_loss/10)*self.transmittance
