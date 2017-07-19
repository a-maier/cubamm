#pragma once

#include "cubamm/integrators/common.hpp"

#include "cuba.h"

namespace cubamm{

  template<>
  class Integrator<divonne> :
    public detail::CommonParameters<Integrator<divonne>>
  {
  public:
    auto & key1(int n){
      key1_ = n;
      return *this;
    }
    auto key1() const{ return key1_; }

    auto & key2(int n){
      key2_ = n;
      return *this;
    }
    auto key2() const{ return key2_; }

    auto & key3(int n){
      key3_ = n;
      return *this;
    }
    auto key3() const{ return key3_; }

    auto & max_pass(int n){
      max_pass_ = n;
      return *this;
    }
    auto max_pass() const{ return max_pass_; }

    auto & border(double d){
      border_ = d;
      return *this;
    }
    auto border() const{ return border_; }

    auto & max_chisq(double d){
      max_chisq_ = d;
      return *this;
    }
    auto max_chisq() const{ return max_chisq_; }

    auto & min_deviation(double d){
      min_deviation_ = d;
      return *this;
    }
    auto min_deviation() const{ return min_deviation_; }

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
    int key1_ = 47;
    int key2_ = 1;
    int key3_ = 1;
    int max_pass_ = 5;
    double border_ = 0.;
    double max_chisq_ = 10.;
    double min_deviation_ = 25.;

    template<typename F, typename... Number>
    auto integrate(
        double /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<double> res;
      Divonne(
          sizeof...(r), 1,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          0, 0, nullptr,
          0, nullptr,
          statefile(), spin(),
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
      Divonne(
          sizeof...(r), N,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          0, 0, nullptr,
          0, nullptr,
          statefile(), spin(),
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
      Divonne(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          0, 0, nullptr,
          0, nullptr,
          statefile(), spin(),
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
      Divonne(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          0, 0, nullptr,
          0, nullptr,
          statefile(), spin(),
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
}
