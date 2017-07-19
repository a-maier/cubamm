#pragma once

#include "cubamm/integrators/common.hpp"

#include "cuba.h"

namespace cubamm{

  template<>
  class Integrator<suave> :
    public detail::CommonParameters<Integrator<suave>>
  {
  public:
    auto & nnew(int n){
      nnew_ = n;
      return *this;
    }
    auto nnew() const{ return nnew_; }

    auto & nmin(int n){
      nmin_ = n;
      return *this;
    }
    auto nmin() const{ return nmin_; }

    auto & flatness(double f){
      flatness_ = f;
      return *this;
    }
    auto flatness() const{ return flatness_; }

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
    int nnew_ = 1000;
    int nmin_ = 2;
    double flatness_ = 50.;

    template<typename F, typename... Number>
    auto integrate(
        double /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<double> res;
      Suave(
          sizeof...(r), 1,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          nnew(), nmin(),
          flatness(), statefile(), spin(),
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
      Suave(
          sizeof...(r), N,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          nnew(), nmin(),
          flatness(), statefile(), spin(),
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
      Suave(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          nnew(), nmin(),
          flatness(), statefile(), spin(),
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
      Suave(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          nnew(), nmin(),
          flatness(), statefile(), spin(),
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
