#pragma once

#include <limits>
#include <memory>
#include <complex>

#include "cubamm/algorithms.hpp"
#include "cubamm/range.hpp"

#include "cuba.h"

namespace cubamm{

  template<class ReturnType>
  struct Result{
    int nregions, neval, fail;
    ReturnType integral, error, prob;

    operator ReturnType(){
      return integral;
    }
  };

  namespace detail{
    template<class Derived>
    class CommonParameters{
    public:
      auto & nvec(int n){
        nvec_ = n;
        return *static_cast<Derived*>(this);
      }
      auto & epsrel(double eps){
        epsrel_ = eps;
        return *static_cast<Derived*>(this);
      }
      auto & epsabs(double eps){
        epsabs_ = eps;
        return *static_cast<Derived*>(this);
      }
      auto & flags(int f){
        flags_ = f;
        return *static_cast<Derived*>(this);
      }
      auto & seed(int s){
        seed_ = s;
        return *static_cast<Derived*>(this);
      }
      auto & mineval(int n){
        mineval_ = n;
        return *static_cast<Derived*>(this);
      }
      auto & maxeval(int n){
        maxeval_ = n;
        return *static_cast<Derived*>(this);
      }
      auto & statefile(std::unique_ptr<char> file){
        statefile_ = std::move(file);
        return *static_cast<Derived*>(this);
      }
      auto & spin(void * s){
        spin_ = s;
        return *static_cast<Derived*>(this);
      }
      auto nvec() const{ return nvec_; }
      auto epsrel() const{ return epsrel_; }
      auto epsabs() const{ return epsabs_; }
      auto flags() const{ return flags_; }
      auto seed() const{ return seed_; }
      auto mineval() const{ return mineval_; }
      auto maxeval() const{ return maxeval_; }
      char const * statefile() const{ return statefile_.get(); }
      auto spin() const{ return spin_; }

    protected:
      CommonParameters() = default;
    private:
      int nvec_ = 1;
      double epsrel_ = 1e-6, epsabs_ = 1e-6;
      int flags_;
      int seed_;
      int mineval_ = 1, maxeval_ = std::numeric_limits<int>::max();
      std::unique_ptr<char> statefile_ = nullptr;
      void * spin_ = nullptr;
    };

    template<typename F, typename T>
    struct IntegrandData{
      F f;
      T arg_ranges;
    };

    template<typename F, typename T>
    IntegrandData<F, T> make_IntegrandData(F f, T t){
      return {std::move(f), std::move(t)};
    }

    template<typename T>
    auto rescale(T t){
      return t;
    }

    template<typename T, typename Number, typename... Numbers>
    auto rescale(T t, Range<Number> r, Range<Numbers>... rest){
      return rescale((r.end - r.start)*t, rest...);
    }

    template<typename T, typename Number, typename... Numbers>
    auto rescale(Result<T> res, Range<Number> r, Range<Numbers>... rest){
      res.integral = rescale(res.integral, r, rest...);
      res.error = rescale(res.error, r, rest...);
      return res;
    }

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
    void copy_to_double_arr(double from, double to[]){
      to[0] = from;
    }

    inline
    void copy_to_double_arr(std::complex<double> const & from, double to[]){
      to[0] = real(from);
      to[1] = imag(from);
    }

    template<size_t N>
    void copy_to_double_arr(std::array<double, N> const & from, double to[]){
      std::copy(begin(from), end(from), to);
    }

    template<size_t N>
    void copy_to_double_arr(
        std::array<std::complex<double>, N> const & from,
        double to[]
    ){
      size_t i = 0;
      for(std::complex<double> const & entry: from){
        to[i++] = real(entry);
        to[i++] = imag(entry);
      }
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

  }

  template<Algorithm a>
  class Integrator;

  template<>
  class Integrator<cuhre> :
    public detail::CommonParameters<Integrator<cuhre>>
  {
  public:
    auto & key(int k){
      key_ = k;
      return *this;
    }
    auto key() const{ return key_; }

    template<typename F, typename... Number>
    auto integrate(F f, Range<Number>... r){
      using ReturnType = decltype(f(r.start...));
      return detail::rescale(
          integrate(ReturnType{}, f, r...),
          r...
      );
    }

    template<typename F, typename... Number>
    auto integrate(
        F f, std::initializer_list<Number>... ranges
    ){
      return integrate(f, detail::to_Range(ranges)...);
    }

  private:
    int key_ = 0;

    template<typename F, typename... Number>
    auto integrate(
        double /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<double> res;
      Cuhre(
          sizeof...(r), 1,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(),
          mineval(), maxeval(),
          key(), statefile(), spin(),
          &res.nregions, &res.neval, &res.fail,
          &res.integral, &res.error, &res.prob
      );
      return res;
    }

    template<size_t N, typename F, typename... Number>
    auto integrate(
        std::array<double, N> /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<std::array<double, N>> res;
      Cuhre(
          sizeof...(r), N,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(),
          mineval(), maxeval(),
          key(), statefile(), spin(),
          &res.nregions, &res.neval, &res.fail,
          res.integral.data(), res.error.data(), res.prob.data()
      );
      return res;
    }

    template<typename F, typename... Number>
    auto integrate(
        std::complex<double> /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<std::complex<double>> res;
      std::array<double, 2> integral, error, prob;
      Cuhre(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(),
          mineval(), maxeval(),
          key(), statefile(), spin(),
          &res.nregions, &res.neval, &res.fail,
          integral.data(), error.data(), prob.data()
      );
      res.integral = std::complex<double>{integral[0], integral[1]};
      res.error = std::complex<double>{error[0], error[1]};
      res.prob = std::complex<double>{prob[0], prob[1]};
      return res;
    }

    template<size_t N, typename F, typename... Number>
    auto integrate(
        std::array<std::complex<double>, N> /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<std::array<std::complex<double>, N>> res;
      std::array<double, 2*N> integral, error, prob;
      Cuhre(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(),
          mineval(), maxeval(),
          key(), statefile(), spin(),
          &res.nregions, &res.neval, &res.fail,
          integral.data(), error.data(), prob.data()
      );
      for(size_t i = 0; i < res.integral.size(); ++i){
        res.integral[i] = std::complex<double>{integral[2*i], integral[2*i+1]};
        res.error[i] = std::complex<double>{error[2*i], error[2*i+1]};
        res.prob[i] = std::complex<double>{prob[2*i], prob[2*i+1]};
      }
      return res;
    }

  };

  template<Algorithm a, typename F, typename... Number>
  auto integrate(F f, Range<Number>... r){
    return Integrator<a>{}.integrate(f, r...);
  }

  template<Algorithm a, typename F, typename... Number>
  auto integrate(
      F f, std::initializer_list<Number>... ranges
  ){
    return integrate<a>(f, detail::to_Range(ranges)...);
  }

}
