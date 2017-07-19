/**
 * @file
 * @brief Header defining integration ranges
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
