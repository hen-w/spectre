// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Zero-initialize the characteristic field and speed Variables tags in
 * the DataBox so they can be filled during time derivative computation and
 * observed. Also default-initialize the InitialBoundaryCharacteristicFields
 * tag to std::nullopt (will be lazily filled on first CRPBC encounter).
 */
struct InitializeCharacteristicObserverTags
    : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<
      Tags::ObserverCharacteristicFieldsTag<3, Frame::Inertial>,
      Tags::ObserverConstraintCharacteristicFieldsTag<3, Frame::Inertial>,
      Tags::ObserverRadiationCharacteristicFieldsTag<3, Frame::Inertial>,
      Tags::ObserverCharacteristicSpeedsTag,
      Tags::ObserverConstraintCharacteristicSpeedsTag,
      Tags::ObserverRadiationCharacteristicSpeedsTag,
      Tags::InitialBoundaryCharacteristicFields<3, Frame::Inertial>>;
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;

  static void apply(
      const gsl::not_null<Variables<Tags::characteristic_fields_tags_list<
          DataVector, 3, Frame::Inertial>>*>
          char_fields,
      const gsl::not_null<
          Variables<Tags::constraint_characteristic_fields_tags_list<
              DataVector, 3, Frame::Inertial>>*>
          constraint_char_fields,
      const gsl::not_null<
          Variables<Tags::radiation_characteristic_fields_tags_list<
              DataVector, 3, Frame::Inertial>>*>
          radiation_char_fields,
      const gsl::not_null<
          Variables<Tags::characteristic_speeds_tags_list>*>
          char_speeds,
      const gsl::not_null<
          Variables<Tags::constraint_characteristic_speeds_tags_list>*>
          constraint_char_speeds,
      const gsl::not_null<
          Variables<Tags::radiation_characteristic_speeds_tags_list>*>
          radiation_char_speeds,
      const gsl::not_null<typename Tags::InitialBoundaryCharacteristicFields<
          3, Frame::Inertial>::type*>
          initial_boundary_char_fields,
      const Scalar<DataVector>& lapse) {
    const size_t num_pts = get(lapse).size();
    char_fields->initialize(num_pts, 0.0);
    constraint_char_fields->initialize(num_pts, 0.0);
    radiation_char_fields->initialize(num_pts, 0.0);
    char_speeds->initialize(num_pts, 0.0);
    constraint_char_speeds->initialize(num_pts, 0.0);
    radiation_char_speeds->initialize(num_pts, 0.0);
    *initial_boundary_char_fields = std::nullopt;
  }
};
}  // namespace Ccz4::fd
