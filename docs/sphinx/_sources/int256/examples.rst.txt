========
Examples
========



Examples of initializing ``avx::Int256``
========================================

``avx::Int256`` can be initialized in the ways listed below:

- Default constructor, all values in vector are set to 0.
- initialization by scalar of type ``int``, all vector fields will be set to the value of passed argument.
- Initialization by another ``avx::Int256``, copy vector value.
- Initialization by ``__m256i``, copy vector value.
- Initialization by initializer list, copy first 8 values to vector. If list contains less than 8 values missing values will be set to 0.
- Initialization by ``std::array<int, 8>`` copy array values to vector.
- Initialization by ``const int*``, copy first 8 values to vector.


.. warning::
    When initializing by ``const int*`` you must ensure that source contains at least 32 bytes of data. 
    Although constructor performs null pointer checks, it can't check for the size of data which is pointed by this pointer.

.. tip::
    Apart from accepting ``std::array<int, 8>`` arrays for smaller types (``short``, ``char``) are also supported.
    Though due to types size difference it will cause performance penalty.

The code snippet below shows various initialization methods in practice.

.. code:: CPP

    avx::Int256 a(1); // This will initialize vector with 1. Printing results of str() method will show Int256(1, 1, 1, 1, 1, 1, 1, 1)
    avx::Int256 b{1, 2, 3, 4, 5}; // Using initializer list. Result will be Int256(1, 2, 3, 4, 5, 0, 0, 0)
    int cP[] = {6, 7, 8, 9, 10, 11, 12, 13}; 
    avx::Int256 c(cP); // Initialization using pointer. Will print Int256(6, 7, 8, 9, 10, 11, 12, 13)

    // For debugginh you can print vector contents by using str() method.
    std::cout << c.str() << '\n';
    // Int256(6, 7, 8, 9, 10, 11, 12, 13)

Usage
======

You can use this type as if it was any other numeric type\*. 

.. important::
    \* Please note that currently ``>``, ``<``, ``>=`` and ``<=`` operators are not suppoerted.

.. code:: CPP

    // Assuming we have set up variables in previous code block.
    avx::Int256 d = c * 3; // Multiplying by scalar.
    std::cout << d.str() << '\n'; // Int256(18, 21, 24, 27, 30, 33, 36, 39)
    b += c; // Adding two vectors together.
    std::cout << b.str() << '\n'; // Int256(7, 9, 11, 13, 15, 11, 12, 13)

    if(a == 1) { // Use == and != operators to quickly check if vectors are equal.
        std::cout << "a vector contains only 1\n";
    }

If you want to save vector contents to external variables, you can do so by using following methods:

- Use ``save`` method to save data to desired destination. 
- Use ``saveAligned`` to save data to memory location aligned to 32-byte boundary.
- Use ``get`` method to retrieve raw vector data (``__m256i``).
- Use ``[]`` operator to access single elements from vector. The vector contents cannot be modified by using this method.

Sample example
==============

The following example shows the most effective way of using ``avx::Int256`` in real computation.

.. code:: CPP

    #include <iostream>
    #include <vector>
    #include <types/int256.hpp>
    
    int main(int argc, char **argv) {
        std::vector aV(1024), bV(1024); // Create two vectors 1024 values each.

        avx::Int256 r; // Create temporary variables.

        size_t i = 0;

        for(; i < aV.size() - avx::Int256::size; i + avx::Int256::size) {
            r = avx::Int256(aV.data() + i) + avx::Int256(bV.data() + i);
            r.save(aV.data() + i);
        }

        for(; i < aV.size(); ++i)
            aV[i] += bV[i];

        return 0;
    }