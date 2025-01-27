========
Examples
========


Example of initializing avx::Char256
====================================

.. code-block:: CPP

    #include <iostream>
    #include <types/char256.hpp>

    int main(int argc, char **argv) {

        avx::Char256 a; // Set internal vector value to 0.
        avx::Char256 b(3); // Initialize vector with value 3.
        avx::Char256 c(std::string("Hello")); // Initialize vector with string "Hello".
        avx::Char256 d({1, 2, 3, 4, 5}); // Initializing using initializer list.
        return 0;
    }


.. important::

    While first two constructors directly initialize underlying vector, the third one will check string size first.
    If size is equal or greater than 32 bytes then first 32 bytes will be copied to the vector. 
    Otherwise temporary buffer will be zeroed and then the string contents will be copied into it.

.. warning::

    ``avx::Char256`` offers another constructor which accepts **explicitly** arguments of ``const char*``. 
    This constructor is only intended to use when loading binary data into vector! 
    Altough it cheks for ``nullptr`` it **does not** check the length of the string.
    It will **ALWAYS** load 32 bytes into vector.

.. todo::
    Differentiate when using `char*` containing characters and binary data (explicittly specify size).


Using basic arithmetic operators
================================

``avx::Char256`` offers two versions of each (non unary) operator.
Let's assume we have created variables:

.. code:: CPP

    avx::Char256 a({1, 2, 3, 4, 5});
    avx::Char256 b({7, 8, 9, 10, 11});
    char c = 5;

One option is to add vectors ``a`` and ``b`` together:

.. code:: cpp
    
    auto d = a + b; // d will have values 8, 10, 12, 14, 16, 0 ... 0

or

.. code:: cpp
    
    a += b; // This time a will have 8, 10, 12, 14, 16, 0 ... 0

However if we want to just add the same value across the vector we can do it like this:

.. code:: cpp
    
    auto d = a + c; // d will have values 6, 7, 8, 9, 10

or

.. code:: cpp
    
    a += c; // same but values will be written to a

List of supported operators
===========================

+------------------+----------------------------------+
|     Operator     |          Supported types         |  
+==================+==================================+
| == [1]_ , != [2]_|  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| +, +=            |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| -, -=            |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| \*, \*=          |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| /, /= [3]_       |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| %, %= [3]_       |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| \|, \|=          |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| &, &=            |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| ^, ^=            |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| <<, <<=          |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| >>, >>= [4]_     |  ``avx::Char256``, ``char``      |
+------------------+----------------------------------+
| ~                |                                  |
+------------------+----------------------------------+


.. [1] Returns ``true`` only if all values are equal (using bitwise XOR).

.. [2] Returns ``true`` if **any** value is not matching (using bitwise XOR).

.. [3] Not available natively in AVX2. Implemented using intermediate ``float`` values.

.. [4] No sign extension.