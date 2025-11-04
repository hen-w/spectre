// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/SpecialMobius.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
using Affine = CoordinateMaps::Affine;
using Equiangular = CoordinateMaps::Equiangular;
using Interval = CoordinateMaps::Interval;

using ad_supported_maps = tmpl::list<
    CoordinateMaps::Affine, CoordinateMaps::ProductOf2Maps<Affine, Affine>,
    CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>,
    CoordinateMaps::BulgedCube, CoordinateMaps::DiscreteRotation<1>,
    CoordinateMaps::DiscreteRotation<2>, CoordinateMaps::DiscreteRotation<3>,
    CoordinateMaps::Equiangular, CoordinateMaps::Frustum,
    CoordinateMaps::Identity<1>, CoordinateMaps::Identity<2>,
    CoordinateMaps::Identity<3>, CoordinateMaps::Interval,
    CoordinateMaps::ProductOf2Maps<Interval, Interval>,
    CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>,
    CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>,
    CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>,
    CoordinateMaps::Rotation<2>, CoordinateMaps::Rotation<3>,
    CoordinateMaps::SpecialMobius, CoordinateMaps::TimeDependent::Rotation<2>,
    CoordinateMaps::TimeDependent::Rotation<3>, CoordinateMaps::Wedge<2>,
    CoordinateMaps::Wedge<3>>;
}  // namespace domain
