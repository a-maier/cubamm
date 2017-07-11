#pragma once

#include <initializer_list>
#include <iterator>
#include <stdexcept>

namespace cubamm{
  template<typename Number>
  struct Range{
    Number start, end;
  };

  namespace detail{
    template<typename Number>
    Range<Number> to_Range(std::initializer_list<Number> range){
      if(range.size() != 2){
        throw std::invalid_argument{
          "initializer list with wrong number of elements"
        };
      }
      return {*begin(range), *std::next(begin(range))};
    }
  }

}
