// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
using Affine = CoordinateMaps::Affine;
using Equiangular = CoordinateMaps::Equiangular;

using ad_supported_maps = tmpl::list<
    CoordinateMaps::Affine, CoordinateMaps::ProductOf2Maps<Affine, Affine>,
    CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>,
    CoordinateMaps::Equiangular,
    CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>,
    CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>,
    CoordinateMaps::Rotation<2>, CoordinateMaps::Rotation<3>,
    CoordinateMaps::Wedge<2>, CoordinateMaps::Wedge<3>>;
}  // namespace domain
