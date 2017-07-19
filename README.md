# cubamm #

cubamm is a C++ wrapper for the [Cuba library](http://www.feynarts.de/cuba/)
for multidimensional numeric integration. The aim is to offer a more
natural and user-friendly interface than Cuba's native C headers.

# Installation #

cubamm is a header-only library, so you can just copy the `cubamm`
folder in `include` to wherever your compiler looks for header files.

Alternatively, you can use [cmake](https://cmake.org/). For this, go to
the source directory and run

``` shell
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/some/path/
make install
```

# Short examples #

Calculate the integral of sqrt(x^2 + y^2) over -1 <= x <= 1,
-2 <= y <= 2 using Cuba's Cuhre algorithm:

``` c++
#include <cmath>
#include "cubamm/cubamm.hpp"

using namespace cubamm;

const double result = integrate<cuhre>(
    [](double x, double y){ return std::sqrt(x*x + y*y); },
    {-1., 1.}, {-2., 2.}
);
```

Complex integration is also supported:

``` c++
#include <cmath>
#include <complex>
#include "cubamm/cubamm.hpp"

using namespace cubamm;
using Complex = std::complex<double>;

const Complex result = integrate<cuhre>(
    [](double x, Complex y){ return std::sqrt(x*x + y*y); },
    {-1., 1.}, {Complex{-2., -2.}, Complex{2., 2.}}
);
```

To change option settings from their defaults construct an integrator
object. This sets the target relative and absolute errors to 10^-8:

``` c++
#include <cmath>
#include "cubamm/cubamm.hpp"

using namespace cubamm;

Integrator<cuhre> i;
i.epsabs(1e-8).epsrel(1e-8);

const double result = i.integrate(
    [](double x, double y){ return std::sqrt(x*x + y*y); },
    {-1., 1.}, {-2., 2.}
);
```
