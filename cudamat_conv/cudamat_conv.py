"""
cudamat style convolution acceleration using a GPU.

Yujia Li, 03/2015
"""

class Tensor4D(object):
    """
    Wrapper of a 4D tensor, this maintains a pointer to a cudamat_4d_tensor
    struct to keep track of both its host and device data.
    """
    def __init__(self, numpy_array=None, n=None, h=None, c=None, w=None):
        pass

