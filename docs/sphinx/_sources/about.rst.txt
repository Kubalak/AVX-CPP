=============
About AVX CPP
=============


| Original idea came after learning about SIMD instruction sets (AVX / AVX2) available on all modern x86 CPUs.
| As there were no libraries directly allowing to use those types of instructions (even though being used by many modern libraries like TensorFlow or Boost) I have decided to create my own.
|     \* Plus to learn more about C++, CMake and low level optimisation ;)

.. note::

    In case of GCC and Clang turning on specific optimization flags might produce code very similar in performance to this wrote by hand.
    Both of these compilers do heavy lifting by integrating SIMD in the output code.
    For MSVC manually optimized SIMD can result in code that is few times faster than "optimized" by MSVC.

.. toctree::
    :maxdepth: 2