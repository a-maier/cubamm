/**
 * @file
 * @brief Divonne integrator
 *
 * This header defines a wrapper around Cuba's Divonne integration
 *
 * @author  Andreas Maier <andreas.maier@durham.ac.uk>
 * @version 0.0.0
 *
 * @section LICENSE
 *
 * Copyright 2017 Andreas Maier
 *
 * This file is part of the cubamm library.
 *
 * The cubamm library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 3 of
 * the License, or (at your option) any later version.
 *
 * The cubamm library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the cubamm library. If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */
#pragma once

#include "cubamm/integrators/common.hpp"

#include "cuba.h"

#include <vector>

#if __cplusplus > 201402L
#include <any>
#else
#include <boost/any.hpp>
#endif

namespace cubamm{
#if __cplusplus > 201402L
  using any = std::any;
  using std::any_cast;
#else
  using any = boost::any;
  using boost::any_cast;
#endif

  namespace detail{
    double rel_point(
        double x, Range<double> const & r
    ){
      return (x - r.start)/(r.end-r.start);
    }

    double rel_point(
        std::complex<double> const & x, Range<std::complex<double>> const & r
    ){
      const std::complex<double> result = (x - r.start)/(r.end-r.start);
      return real(result);
    }

    template<typename... T, size_t... indices>
    std::array<double, sizeof...(T)> to_cuba_coordinates(
        std::tuple<T...> const & point,
        std::tuple<Range<T>...> const & transform_range,
        std::index_sequence<indices...>
    ){
      return {
        rel_point(
            std::get<indices>(point), std::get<indices>(transform_range)
        )...
      };
    }

    template<typename... T>
    std::array<double, sizeof...(T)> to_cuba_coordinates(
        std::tuple<T...> const & point,
        std::tuple<Range<T>...> const & transform_range
    ){
      static_assert(point.size() == transform_range.size(), "compatible size");
      return point_to_cuba_coordinates(
          point, transform_range, std::index_sequence_for<T...>{}
      );
    }

    template<typename... T>
    auto to_cuba_coordinates(
        std::vector<any> const & points,
        std::tuple<Range<T>...> const & transform_ranges
    ){
      using CubaCoordinate = std::array<double, sizeof...(T)>;
      std::vector<CubaCoordinate> cuba_coordinates(points.size());
      std::transform(
          begin(points), end(points),
          begin(cuba_coordinates),
          [transform_ranges](any const & point){
            return point_to_cuba_coordinates(
                any_cast<std::tuple<T...>>(point),
                transform_ranges
            );
          }
      );
      return cuba_coordinates;
    }

  }

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

    template<typename... T>
    auto & given(std::vector<std::tuple<T...>> t){
      known_peaks_ = std::move(t);
      return *this;
    }

    template<typename InputIterator>
    auto & given(InputIterator first, InputIterator last){
      known_peaks_.assign(first, last);
      return *this;
    }

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
    std::vector<any> known_peaks_;

    template<typename F, typename... Number>
    auto integrate(
        double /* function f return type */,
        F f, Range<Number>... r
    ){
      auto data = detail::make_IntegrandData(std::move(f), std::make_tuple(r...));
      Result<double> res;
      const auto peaks = detail::to_cuba_coordinates(
          known_peaks_, data.ranges
      );
      Divonne(
          sizeof...(r), 1,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          peaks.size(), sizeof...(r), peaks.data(),
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
      const auto peaks = detail::to_cuba_coordinates(
          known_peaks_, data.ranges
      );
      Divonne(
          sizeof...(r), N,
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          peaks.size(), sizeof...(r), peaks.data(),
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
      const auto peaks = detail::to_cuba_coordinates(
          known_peaks_, data.ranges
      );
      Divonne(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          peaks.size(), sizeof...(r), peaks.data(),
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
      const auto peaks = detail::to_cuba_coordinates(
          known_peaks_, data.ranges
      );
      Divonne(
          sizeof...(r), integral.size(),
          detail::as_cuba_integrand<decltype(data)>, &data, nvec(),
          epsrel(), epsabs(),
          flags(), seed(),
          mineval(), maxeval(),
          key1(), key2(), key3(),
          max_pass(), border(),
          max_chisq(), min_deviation(),
          peaks.size(), sizeof...(r), peaks.data(),
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
