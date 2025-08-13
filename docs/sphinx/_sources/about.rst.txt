=============
About AVX CPP
=============


| Original idea came after learning about SIMD instruction sets (AVX / AVX2) available on all modern x86 CPUs.
| As there were no libraries directly allowing to use those types of instructions (even though being used by many modern libraries like TensorFlow, Boost etc.) I have decided to create my own (cause why not).
|     \* Plus to learn more about C++, CMake and low level optimisation ;)

.. note::

    In case of GCC and Clang turning on specific optimization flags might produce code very similar in performance to this wrote by hand.
    Both of these compilers can produce well optimized code when SIMD instructions are enabled.
    For MSVC manually optimized SIMD can result in code that is few times faster than "optimized" by MSVC.

While developing this library I have followed the principles listed below:

- **Performance** - the library should be as fast as possible, resulting in code that has as little overhead caused by abstraction as possible (that's why almost all methods are `inline`). 
- **Exception Avoidance** - the library should not throw exceptions as they make code slower and harder to read. An exception to this rule is running code in Debug mode, where exceptions are explicitly thrown. Otherwise, for *invalid* inputs, functions have no effect or guarantee not to cause undefined behavior (e.g. buffer overflow).
- **Ease of Use** - the library should be easy to use providing intuitive API for introducing SIMD to existing code.
- **Simplicity** - no one likes to use overcomplicated code with many abstraction layers. That's why each class is an independent unit (even though all of them share same namespace and most methods) not using polymorphism or iheritance.
- **Portability** - the library should be available on Windows and Linux platforms while supporting all major compilers (GCC, Clang and MSVC). For obvious reasons the library is targeting only x86 architecture supporting AVX2 (mandatory) and AVX512 (optional).
- `TDD <https://www.geeksforgeeks.org/test-driven-development-tdd/>`__ - each feature should be tested before being added to the library (along with adding a feature, tests for it should be developed). The tests should be run on all supported platforms and compilers to ensure seamless integration [1]_. Tests can be found in `src/tests <https://github.com/Kubalak/AVX-CPP/tree/main/src/tests>`__ folder.

List of classes
===============

The library provides following classes, which can be used (all of which are within ``avx`` namespace):

- ``Char256`` holds 32 chars (8-bit integers),
- ``UChar256`` holds 32 unsigned chars (8-bit unsigned integers),
- ``Short256`` holds 16 shorts (16-bit integers),
- ``UShort256`` holds 16 unsigned shorts (16-bit unsigned integers),
- ``Int256`` holds 8 32-bit integers,
- ``UInt256`` holds 8 32-bit unsigned integers,
- ``Long256`` holds 4 64-bit integers,
- ``ULong256`` holds 4 64-bit unsigned integers,
- ``Float256`` holds 8 32-bit floats,
- ``Double256`` holds 4 64-bit doubles

| Almost all classes are header only and utilize inline methods for better performance.
| Please refer to `this <https://en.wikipedia.org/wiki/Inline_(C_and_C%2B%2B)>`__ page for more information what benefits does inlining provide and what are it's drawbacks.

Features
========

Each of the types supports regular math operators:

- ``+`` and ``+=``
- ``-`` and ``-=``
- ``*`` and ``*=``
- ``/`` and ``/=``

Each operator supports class and scalar values (of corresponding type).
For example ``+`` operator for ``Int256`` will have two versions, one accepting ``Int256`` and other accepting ``int``.

All types also support following logical operators:

- ``==``, accepts types like math operators,
- ``!=``, same as above + ignores -0.0 for floating-point types.

Integer types also support bitwise operators:

- ``&`` and ``&=``
- ``|`` and ``|=``
- ``^`` and ``^=``
- ``~`` without arguments
- ``<<`` and ``<<=``, this is an exception, accepts same class or ``unsigned int``
- ``>>`` and ``>>=``, this is an exception, accepts same class or ``unsigned int``

Each class will provide list of operators and used SIMD instructions.

Limitations
===========

| Currently the library does not support dynamic detection of supported SIMD instructions.
| This means that if it is compiled with AVX512 enabled, then it will only work on CPUs supporting AVX512.
| For ``Long256`` and ``ULong256`` data loading and saving overhead results in SIMD version being slower, than regular sequential solution.

.. note::

    Please check `Long256 <long256/index.html>`__ and `ULong256 <ulong256/index.html>`__ documentation to see details.

Each constructor or ``load`` method expects source to contain at least 32 bytes of continous data.
Providing pointer to memory, which contains less than 32 bytes can result in undefined behavior.

.. toctree::
    :maxdepth: 2

----------------------

.. [1] Automatic tests are run using Github Actions on each commit. As some functionality is not available on all platforms (e.g. SVML) some features are fully implemented on selected platforms and being ported to others later (e.g. stable `/` operator for `Int256`).
