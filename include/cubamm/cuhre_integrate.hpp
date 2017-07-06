#pragma once

#include <array>

namespace cubamm{
  struct Range{
    double start, end;
  };

  namespace detail{
    template<typename F, typename T, size_t N, size_t... indices>
    auto apply_unpacked(
        F f, std::array<T, N> const & args,
        std::index_sequence<indices...>
    ){
      return f(args[indices]...);
    }

    template<typename F, typename T, size_t N>
    auto apply_unpacked(F&& f, std::array<T, N> const & args){
      return apply_unpacked(
          std::forward<F>(f), args,
          std::make_index_sequence<N>{}
      );
    }

    inline
    void copy_to_double_arr(double in, double to[]){
      to[0] = in;
    }

    template<typename F, size_t N>
    struct IntegrandData{
      F f;
      std::array<Range, N> arg_ranges;
    };

    template<typename F, size_t N>
    int as_cuba_integrand(
        const int * /* ndim */, const double x[],
        const int * /* ncomp */, double f[],
        void * userdata
    ){
      auto data = static_cast<IntegrandData<F,N>*>(userdata);
      auto const & ranges = data->arg_ranges;
      std::array<double, N> args;
      for(size_t i = 0; i < N; ++i){
        args[i] = (ranges[i].end - ranges[i].start)*x[i] + ranges[i].start;
      }
      const auto res = apply_unpacked(data->f, args);
      copy_to_double_arr(res, f);
      return 0;
    }

    using integrand_t = int (
        const int*, const double[], const int *, double[], void*
    );

    double cuhre_integrate(integrand_t, void * data);

    template<typename T>
    auto rescale(T t){
      return t;
    }

    template<typename T, typename... Ranges>
    auto rescale(T t, Range r, Ranges&&... rest){
      return rescale((r.end - r.start)*t, rest...);
    }
  }

  template<typename F, typename... Ranges>
  double cuhre_integrate(F f, Ranges... r){
    constexpr size_t N = sizeof...(r);
    detail::IntegrandData<F,N> data{std::move(f), {r...}};
    return detail::rescale(
        detail::cuhre_integrate(detail::as_cuba_integrand<F,N>, &data),
        r...
    );
  }

}
