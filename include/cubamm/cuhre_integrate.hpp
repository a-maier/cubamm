#pragma once

#include <tuple>
#include <type_traits>

namespace cubamm{
  template<typename Number>
  struct Range{
    Number start, end;
  };

  namespace detail{
    template<typename F, typename... T, size_t... indices>
    auto apply_unpacked(
        F f, std::tuple<T...> const & args,
        std::index_sequence<indices...>
    ){
      return f(std::get<indices>(args)...);
    }

    template<typename F, typename... T>
    auto apply_unpacked(F&& f, std::tuple<T...> const & args){
      return apply_unpacked(
          std::forward<F>(f), args,
          std::index_sequence_for<T...>{}
      );
    }

    inline
    void copy_to_double_arr(double in, double to[]){
      to[0] = in;
    }

    template<typename F, typename Tuple>
    struct IntegrandData{
      F f;
      Tuple arg_ranges;
    };

    template<typename F, typename Tuple>
    IntegrandData<F, Tuple> make_IntegrandData(F f, Tuple t){
      return {std::move(f), std::move(t)};
    }

    template<typename Result, typename... T>
    void transform_args(
        const double *, std::tuple<T...> const &,
        Result &,
        std::index_sequence<>
    ){}

    template<typename Result, typename... T, size_t I, size_t... U>
    void transform_args(
        const double args[], std::tuple<T...> const & transforms,
        Result & result,
        std::index_sequence<I, U...> /* index */
    ){
      static_assert(I < sizeof...(T), "index too large!");
      auto const & range = std::get<I>(transforms);
      std::get<I>(result) = (range.end - range.start)*args[I] + range.start;
      transform_args(args, transforms, result, std::index_sequence<U...>{});
    }

    template<typename... T>
    std::tuple<T...> transform_args(
        const double args[], std::tuple<Range<T>...> const & transforms
    ){
      std::tuple<decltype((T{}-T{})*0.+T{})...> result{};
      transform_args(
          args, transforms, result,
          std::index_sequence_for<T...>{}
      );
      return result;
    }

    template<typename Integrand>
    int as_cuba_integrand(
        const int * /* ndim */, const double x[],
        const int * /* ncomp */, double f[],
        void * userdata
    ){
      auto integrand = static_cast<Integrand*>(userdata);
      auto const & ranges = integrand->arg_ranges;
      auto args = transform_args(x, ranges);
      const auto res = apply_unpacked(integrand->f, args);
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

    template<typename T, typename Number, typename... Numbers>
    auto rescale(T t, Range<Number> r, Range<Numbers>... rest){
      return rescale((r.end - r.start)*t, rest...);
    }

    template<typename Number>
    Range<Number> to_Range(std::initializer_list<Number> range){
      return {*begin(range), *std::next(begin(range))};
    }
  }

  template<typename F, typename... Number>
  double cuhre_integrate(F f, Range<Number>... r){
    auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
    return detail::rescale(
        detail::cuhre_integrate(
            detail::as_cuba_integrand<decltype(data)>, &data
        ),
        r...
    );
  }

  template<typename F, typename... Number>
  double cuhre_integrate(F f, std::initializer_list<Number>... ranges){
    return cuhre_integrate(f, detail::to_Range(ranges)...);
  }

}
