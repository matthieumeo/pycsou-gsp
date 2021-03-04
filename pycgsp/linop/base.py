# #############################################################################
# base.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Base class for polynomial graph operators. Useful for implementing graph convolutions and generalised graph differential operators.
"""
from numbers import Number
from typing import Union

import numpy as np

from pycsou.core.linop import LinearOperator


class PolynomialLinearOperator(LinearOperator):
    r"""
    Polynomial linear operator :math:`P(L)`.

    Given a polynomial :math:`P(x)=\sum_{k=0}^N a_k x^k` and a square linear operator :math:`\mathbf{L}:\mathbb{R}^N\to \mathbb{R}^N,`
    we define the polynomial linear operator :math:`P(\mathbf{L}):\mathbb{R}^N\to \mathbb{R}^N` as:

    .. math::

       P(\mathbf{L})=\sum_{k=0}^N a_k \mathbf{L}^k,

    where :math:`\mathbf{L}^0` is the identity matrix.
    The *adjoint* of :math:`P(\mathbf{L})` is given by:

    .. math::

       P(\mathbf{L})^\ast=\sum_{k=0}^N a_k (\mathbf{L}^\ast)^k.

    Examples
    --------

    .. testsetup::

       import numpy as np

    .. doctest::

       >>> from pycsou.linop import DenseLinearOperator
       >>> from pycgsp.linop.base import PolynomialLinearOperator
       >>> L = DenseLinearOperator(np.arange(64).reshape(8,8))
       >>> PL = PolynomialLinearOperator(LinOp=L, coeffs=[1/2 ,2, 1])
       >>> x = np.arange(8)
       >>> np.allclose(PL(x), x/2 + 2 * L(x) + (L**2)(x))
       True

    """

    def __init__(self, LinOp: LinearOperator, coeffs: Union[np.ndarray, list, tuple]):
        r"""

        Parameters
        ----------
        LinOp: pycsou.core.LinearOperator
            Square linear operator :math:`\mathbf{L}`.
        coeffs: Union[np.ndarray, list, tuple]
            Coefficients :math:`\{a_0,\ldots, a_N\}` of the polynomial :math:`P`.
        """
        self.coeffs = np.asarray(coeffs).astype(LinOp.dtype)
        if LinOp.shape[0] != LinOp.shape[1]:
            raise ValueError('Input linear operator must be square.')
        else:
            self.Linop = LinOp
        super(PolynomialLinearOperator, self).__init__(shape=LinOp.shape, dtype=LinOp.dtype,
                                                       is_explicit=LinOp.is_explicit, is_dense=LinOp.is_dense,
                                                       is_sparse=LinOp.is_sparse,
                                                       is_dask=LinOp.is_dask,
                                                       is_symmetric=LinOp.is_symmetric)

    def __call__(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        z = x.astype(self.dtype)
        y = self.coeffs[0] * x
        for i in range(1, len(self.coeffs)):
            z = self.Linop(z)
            y += self.coeffs[i] * z
        return y

    def adjoint(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if self.is_symmetric:
            return self(x)
        else:
            z = x.astype(self.dtype)
            y = np.conj(self.coeffs[0]) * x
            for i in range(1, len(self.coeffs)):
                z = self.Linop.adjoint(z)
                y += np.conj(self.coeffs[i]) * z
            return y
