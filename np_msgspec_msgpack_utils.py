#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Line_Of_Sight
File   : np_msgspec_msgpack_utils.py

Author: Pessel Arnaud
Date: 2025-05-02
Version: 1.0
GitHub: https://github.com/dunaar/Line_Of_Sight
License: MIT

Description:
    This module provides utilities for serializing and deserializing complex Python objects
    (e.g., NumPy arrays, Numba typed lists, and complex numbers) using the msgspec library
    with MessagePack encoding. It defines custom encoding and decoding hooks to handle
    specific data types that are not natively supported by MessagePack, enabling efficient
    data serialization for the Line_Of_Sight project.

    Key Features:
    - Custom serialization for NumPy arrays, scalars, Numba typed lists, and complex numbers.
    - Uses extension type codes to differentiate between object types during encoding/decoding.
    - Fallback to standard Python lists if Numba is unavailable.
    - Includes a test suite in the __main__ block to verify serialization and deserialization.

Dependencies:
    - Python 3.x
    - msgspec: For efficient MessagePack serialization and deserialization.
    - NumPy: For handling array and scalar data types.
    - Numba (optional): For typed list optimization.
    - struct: For packing/unpacking complex numbers.

Usage:
    This module is typically imported to provide `enc` and `dec` objects for encoding/decoding
    data structures in the Line_Of_Sight project. The `__main__` block can be run to test
    serialization and deserialization of various data types.

    Example:
    ```bash
    python np_msgspec_msgpack_utils.py
    ```
"""

__version__ = "1.0"

# === Built-in ===
import logging
import struct
from typing import Any, Union, List, Dict

# === Third-party ===
import msgspec
import numpy as np
try:
    from numba.typed import List as Numba_List  # type: ignore # Numba typed list for performance optimization
except ImportError:
    logging.warning('ImportError: Normal list is used instead of Numba list')
    Numba_List = list  # type: ignore

# All extension types need a unique integer designator so the decoder knows
# which type they're decoding. Here we arbitrarily choose 1, but any integer
# between 0 and 127 (inclusive) would work.
COMPLEX_TYPE_CODE =  1
NUMBA_LIST_CODE   =  2
NUMPY_ND_CODE     = 10
NUMPY_NB_CODE     = 11

def enc_hook(obj: Any) -> Any:
    """
    Encode hook for custom serialization of specific object types using msgspec.

    Parameters:
    - obj: The object to be serialized (Any).

    Returns:
    - The serialized object as a msgspec.msgpack.Ext object or raises an error for unsupported types.

    Raises:
    - NotImplementedError: If the object type is not supported for serialization.

    Notes:
    - Handles Numba typed lists, NumPy arrays, NumPy scalars, and complex numbers.
    - Converts objects to a format suitable for MessagePack serialization with extension codes.
    - NumPy arrays are serialized with their dtype, shape, and data buffer.
    - Complex numbers are packed into a binary format using struct.
    """
    if isinstance(obj, Numba_List):
        # If the object is a Numba typed list, we convert it to a regular list
        # and pack it with a specific code.
        data = msgspec.msgpack.encode(list(obj))
        return msgspec.msgpack.Ext(NUMBA_LIST_CODE, data)
    elif isinstance(obj, np.ndarray):
        # If the dtype is structured, store the interface description;
        data = msgspec.msgpack.encode({b'dtype': obj.dtype.str, b'shape': obj.shape, b'data': obj.tobytes()})
        return msgspec.msgpack.Ext(NUMPY_ND_CODE, data)
    elif isinstance(obj, (np.bool_, np.number)):
        data = msgspec.msgpack.encode({b'dtype': obj.dtype.str, b'data': obj.tobytes()})
        return msgspec.msgpack.Ext(NUMPY_NB_CODE, data)
    elif isinstance(obj, complex):
        data = struct.pack('dd', obj.real, obj.imag)
        return msgspec.msgpack.Ext(COMPLEX_TYPE_CODE, data)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(f"Objects of type {type(obj)} are not supported")

def ext_hook(code: int, data: memoryview) -> Any:
    """
    Decode hook for custom deserialization of MessagePack extension types.

    Parameters:
    - code: The extension type code (int).
    - data: The raw data buffer (memoryview).

    Returns:
    - The deserialized object (Numba_List, np.ndarray, np scalar, or complex).

    Raises:
    - NotImplementedError: If the extension type code is not supported.

    Notes:
    - Decodes Numba typed lists, NumPy arrays, NumPy scalars, and complex numbers.
    - Uses extension type codes to identify the object type during deserialization.
    - NumPy arrays are reconstructed from their dtype, shape, and data buffer.
    - Complex numbers are unpacked from binary format using struct.
    """
    if code == NUMBA_LIST_CODE:
        # This extension type represents a Numba typed list, decode the data
        # buffer and return a Numba typed list.
        data = msgspec.msgpack.decode(data)
        return Numba_List(data)
    elif code == NUMPY_ND_CODE:
        # This extension type represents a NumPy ndarray, decode the data
        # buffer accordingly.
        data = msgspec.msgpack.decode(data)
        return np.frombuffer(data[b'data'], dtype=np.dtype(data[b'dtype'])).reshape(data[b'shape'])
    elif code == NUMPY_NB_CODE:
        # This extension type represents a NumPy scalar, decode the data
        # buffer accordingly.
        data = msgspec.msgpack.decode(data)
        return np.frombuffer(data[b'data'], dtype=np.dtype(data[b'dtype']))[0]
    elif code == COMPLEX_TYPE_CODE:
        # This extension type represents a complex number, decode the data
        # buffer accordingly.
        real, imag = struct.unpack('dd', data)
        return complex(real, imag)
    else:
        # Raise a NotImplementedError for other extension type codes
        raise NotImplementedError(f"Extension type code {code} is not supported")


# Create an encoder and a decoder using the custom callbacks
enc = msgspec.msgpack.Encoder(enc_hook=enc_hook)
dec = msgspec.msgpack.Decoder(ext_hook=ext_hook)

if __name__ == '__main__':
    """
    Main function to test serialization and deserialization of various data types.

    Description:
        This block defines a test message containing a variety of Python and NumPy
        data types, including lists, dictionaries, NumPy arrays, Numba typed lists,
        and complex numbers. It encodes the message using the custom encoder and
        decodes it to verify that the original data is preserved. The test results
        are printed to the console.

    Notes:
        - Requires the `sys` and `tempfile` modules, which are imported locally to
          adhere to import preservation rules.
        - The test message includes a mix of standard Python types and custom types
          handled by the `enc_hook` and `ext_hook` functions.
        - Logging is configured to output informational messages to the console.
    """
    # === Built-in ===
    import sys
    import tempfile

    # Configure logging for the test suite
    logging.basicConfig(
        level=logging.INFO,  # Set logging level to INFO
        format='%(asctime)s-%(levelname)s-%(module)s-%(funcName)s: %(message)s',  # Define log message format
        handlers=[
            logging.StreamHandler()  # Add console handler for logging
        ]
    )

    # Define a test message with various data types
    msg = {
        'list':       [1, 2, 3, 4],                             # Standard Python list
        'dict':       {'a': 1, 'b': 2, 'c': 3},                 # Standard Python dictionary
        'tuple':      (1, 2, 3, 4),                             # Standard Python tuple
        'set':        {1, 2, 3, 4},                             # Standard Python set
        'frozenset':  frozenset([1, 2, 3, 4]),                  # Standard Python frozenset
        'bytes':      b'hello',                                 # Bytes object
        'bytearray':  bytearray(b'world'),                      # Bytearray object
        'str':        'hello world',                            # String
        'int':        42,                                       # Integer
        'float':      3.14,                                     # Float
        'none':       None,                                     # NoneType
        'complex':    complex(1, 2),                            # Complex number
        'cplx_list':  [0, 0.75, 1 + 0.5j, 1 - 0.5j],            # List with complex numbers
        'cplx_tuple': (0, 0.75, 1 + 0.5j, 1 - 0.5j),            # Tuple with complex numbers
        'cplx':       [0, 0.75, 1 + 0.5j, 1 - 0.5j],            # Duplicate list for testing
        'nb_list':    Numba_List([1, 2, 3, 4]),                 # Numba typed list
        'nd_array':   np.array([[1, 2], [3, 4]], dtype=np.float32),  # NumPy 2D array
        'nb_array':   np.array([1, 2, 3], dtype=np.int32),           # NumPy 1D array
        'bool_array': np.array([True, False, True], dtype=np.bool_), # NumPy boolean array
        'str_array':  np.array(['a', 'b', 'c'], dtype='U1'),    # NumPy string array
        'bool_scalar': np.bool_(True),                          # NumPy boolean scalar
        'nb_scalar':  np.int32(42),                             # NumPy integer scalar
        'nd_scalar':  np.float64(3.14)                          # NumPy float scalar
    }

    # Encode and decode the message to verify serialization
    buf  = enc.encode(msg)  # Serialize the test message
    msg2 = dec.decode(buf)  # Deserialize the message
    print(msg2)             # Output the deserialized message to verify correctness