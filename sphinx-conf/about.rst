=============
About AVX CPP
=============


| Original idea came after learning about SIMD instruction sets (AVX / AVX2) available on all modern x86 CPUs.
| As there were no libraries directly allowing to use those types of instructions (even though being used by many modern libraries like TensorFlow, Boost etc.) I have decided to create my own (cause why not).
|     \* Plus to learn more about C++, CMake and low level optimisation ;)

.. note::

    In case of GCC and Clang turning on specific optimization flags might produce code very similar in performance to this wrote by hand.
    Both of these compilers do heavy lifting by integrating SIMD in the output code.
    For MSVC manually optimized SIMD can result in code that is few times faster than "optimized" by MSVC.

While developing this library I have followed the principles listed below:

- **Performance** - the library should be as fast as possible, resulting in code that has as little overhead caused by abstraction as possible (that's why almost all methods are `inline`). 
- **Exception Avoidance** - the library should not throw exceptions as they make code slower and harder to read. An exception to this rule is running code in Debug mode, where exceptions are explicitly thrown. Otherwise, for _invalid_ inputs, functions have no effect or guarantee not to cause undefined behavior (e.g. buffer overflow).
- **Ease of Use** - the library should be easy to use providing intuitive API for introducing SIMD to existing code.
- **Simplicity** - no one likes to use overcomplicated code with many abstraction layers. That's why each class is an independent unit (even though all of them share same namespace and most methods) not using polymorphism or iheritance.
- **Portability** - the library should be available on Windows and Linux platforms while supporting all major compilers (GCC, Clang and MSVC). For obvious reasons the library is targeting only x86 architecture supporting AVX2 (mandatory) and AVX512 (optional).
- `TDD <https://www.geeksforgeeks.org/test-driven-development-tdd/>`__ - each feature should be tested before being added to the library (along with adding a feature, tests for it should be developed). The tests should be run on all supported platforms and compilers to ensure seamless integration [1]_. 

.. toctree::
    :maxdepth: 2

----------------------

.. [1] Automatic tests are run using Github Actions on each commit. As some functionality is not available on all platforms (e.g. SVML) some features are fully implemented on selected platforms and being ported to others later (e.g. stable `/` operator for `Int256`).
