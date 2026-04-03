// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/ConstraintsRadiationPreserving.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Ccz4::BoundaryConditions::BoundaryCondition,
        tmpl::list<Ccz4::BoundaryConditions::ConstraintsRadiationPreserving>>>;
  };
};

SPECTRE_TEST_CASE("Unit.Ccz4.BoundaryConditions.ConstraintsRadiationPreserving",
                  "[Unit][Evolution]") {
  register_factory_classes_with_charm<Metavariables>();
  {
    INFO("Test creation and serialization");
    const auto boundary_condition =
        TestHelpers::test_creation<
            std::unique_ptr<Ccz4::BoundaryConditions::BoundaryCondition>,
            Metavariables>("ConstraintsRadiationPreserving:")
            ->get_clone();

    const auto serialized_and_deserialized_condition =
        serialize_and_deserialize(
            *dynamic_cast<
                Ccz4::BoundaryConditions::ConstraintsRadiationPreserving*>(
                boundary_condition.get()));
    CHECK(serialized_and_deserialized_condition.get_clone() != nullptr);
  }
  {
    INFO("Test bc_type is TimeDerivative");
    CHECK(Ccz4::BoundaryConditions::ConstraintsRadiationPreserving::bc_type ==
          evolution::BoundaryConditions::Type::TimeDerivative);
  }
}
}  // namespace
