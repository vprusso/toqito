state_props.is_unextendible_product_basis
=========================================

.. py:module:: state_props.is_unextendible_product_basis

.. autoapi-nested-parse::

   Check if a set of states form an unextendible product basis.



Functions
---------

.. autoapisummary::

   state_props.is_unextendible_product_basis.is_unextendible_product_basis


Module Contents
---------------

.. py:function:: is_unextendible_product_basis(vecs, dims)

   Check if a set of vectors form an unextendible product basis (UPB) :cite:`Bennett_1999_UPB`.

   Consider a multipartite quantum system :math:`\mathcal{H} = \bigotimes_{i=1}^{m} \mathcal{H}_{i}` with :math:`m`
   parties with respective dimensions :math:`d_i, i = 1, 2, ..., m`. An (incomplete orthogonal) product basis (PB) is a
   set :math:`S` of pure orthogonal product states spanning a proper subspace :math:`\mathcal{H}_S` of
   :math:`\mathcal{H}`.  An unextendible product basis (UPB) is a PB whose complementary subspace
   :math:`\mathcal{H}_S-\mathcal{H}` contains no product state.  This function is inspired from `IsUPB` in
   :cite:`QETLAB_link`.

   .. rubric:: Examples

   See :func:`.tile`. All the states together form a UPB:

   >>> import numpy as np
   >>> from toqito.states import tile
   >>> from toqito.state_props import is_unextendible_product_basis
   >>>
   >>> upb_tiles = np.array([tile(i) for i in range(5)])
   >>> dims = np.array([3, 3])
   >>> is_unextendible_product_basis(upb_tiles, dims)
   (True, None)

   However, the first 4 do not:

   >>> import numpy as np
   >>> from toqito.states import tile
   >>> from toqito.state_props import is_unextendible_product_basis
   >>>
   >>> non_upb_tiles = np.array([tile(i) for i in range(4)])
   >>> dims = np.array([3, 3])
   >>> is_unextendible_product_basis(non_upb_tiles, dims)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
   (False, array([-0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
       0.00000000e+00,  0.00000000e+00, -1.11022302e-16,  7.07106781e-01,
       7.07106781e-01]))

   The orthogonal state is given by

   .. math::
       \frac{1}{\sqrt{2}} |2\rangle \left( |1\rangle + |2\rangle \right)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If product of dimensions does not match the size of a vector.
   :raises ValueError: If at least one vector is not a product state.
   :param vecs: The list of states.
   :param dims: The list of dimensions.
   :return: Returns a tuple. The first element is :code:`True` if input is a UPB and :code:`False` otherwise. The
            second element is a witness (a product state orthogonal to all the input vectors) if the input is a
            PB and :code:`None` otherwise.


