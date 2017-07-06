#include "cubamm/cuhre_integrate.hpp"
#include <limits>

#include "cuba.h"

namespace cubamm{

  namespace detail{
    namespace{
      constexpr int n_args = 2;
      constexpr int dim_result = 1;
      constexpr int n_vec = 1;
      constexpr double err = 1e-6;
      constexpr int flags = 0;
      constexpr int min_eval = 10;
      constexpr auto max_eval = std::numeric_limits<int>::max();
      constexpr int key = 0;
      constexpr char* statefile = nullptr;
      constexpr char* spin = nullptr;

      struct CubaResult{
        int nregions, neval, fail;
        std::array<double, dim_result> integral, error, prob;
      };
    }

    double cuhre_integrate(integrand_t integrand, void * data){
      CubaResult result;
      Cuhre(
          n_args, dim_result, integrand, data, n_vec,
          err, err, flags,
          min_eval, max_eval,
          key,
          statefile, spin,
          &result.nregions, &result.neval, &result.fail,
          result.integral.data(), result.error.data(), result.prob.data()
      );
      return result.integral[0];
    }
  }
}
