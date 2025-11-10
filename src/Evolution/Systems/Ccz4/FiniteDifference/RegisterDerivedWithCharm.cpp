// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/RegisterDerivedWithCharm.hpp"

#include "Evolution/Systems/Ccz4/FiniteDifference/DummyReconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace Ccz4::fd {
void register_derived_with_charm() {
  register_classes_with_charm(typename Reconstructor::creatable_classes{});
}
}  // namespace Ccz4::fd
