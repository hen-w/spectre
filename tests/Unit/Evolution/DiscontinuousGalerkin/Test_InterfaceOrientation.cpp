// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceOrientation.hpp"
#include "Utilities/GetOutput.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
Neighbors<Dim> make_neighbors(const ElementId<Dim>& neighbor_id,
                              const OrientationMap<Dim>& orientation) {
  return Neighbors<Dim>{std::unordered_set<ElementId<Dim>>{neighbor_id},
                        orientation};
}

void test_aligned_1d() {
  // Two aligned elements sharing an interface in the xi direction. The lower
  // element looks upper_xi at the upper element, and vice versa; the two must
  // reach opposite conclusions.
  const ElementId<1> lower_element{0, {{SegmentId{2, 0}}}};
  const ElementId<1> upper_element{0, {{SegmentId{2, 1}}}};
  const auto aligned = OrientationMap<1>::create_aligned();

  // From the lower element, looking upper_xi at the upper element: the lower
  // element is the Lower element of the interface.
  CHECK(interface_orientation<1>(lower_element, Direction<1>::upper_xi(),
                                 make_neighbors<1>(upper_element, aligned),
                                 upper_element) ==
        InterfaceOrientation::InteriorIsLower);
  // From the upper element, looking lower_xi at the lower element: the upper
  // element is the Upper element of the interface. Antisymmetry.
  CHECK(interface_orientation<1>(upper_element, Direction<1>::lower_xi(),
                                 make_neighbors<1>(lower_element, aligned),
                                 lower_element) ==
        InterfaceOrientation::InteriorIsUpper);
}

void test_aligned_2d() {
  const ElementId<2> lower_element{0, {{SegmentId{2, 0}, SegmentId{1, 0}}}};
  const ElementId<2> upper_element{0, {{SegmentId{2, 1}, SegmentId{1, 0}}}};
  const auto aligned = OrientationMap<2>::create_aligned();

  CHECK(interface_orientation<2>(lower_element, Direction<2>::upper_xi(),
                                 make_neighbors<2>(upper_element, aligned),
                                 upper_element) ==
        InterfaceOrientation::InteriorIsLower);
  CHECK(interface_orientation<2>(upper_element, Direction<2>::lower_xi(),
                                 make_neighbors<2>(lower_element, aligned),
                                 lower_element) ==
        InterfaceOrientation::InteriorIsUpper);
}

void test_aligned_3d() {
  const ElementId<3> lower_element{
      0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{2, 0}}}};
  const ElementId<3> upper_element{
      0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{2, 1}}}};
  const auto aligned = OrientationMap<3>::create_aligned();

  CHECK(interface_orientation<3>(lower_element, Direction<3>::upper_zeta(),
                                 make_neighbors<3>(upper_element, aligned),
                                 upper_element) ==
        InterfaceOrientation::InteriorIsLower);
  CHECK(interface_orientation<3>(upper_element, Direction<3>::lower_zeta(),
                                 make_neighbors<3>(lower_element, aligned),
                                 lower_element) ==
        InterfaceOrientation::InteriorIsUpper);
}

void test_twisted_2d() {
  // An inter-block interface in the xi direction whose OrientationMap reverses
  // the xi axis. Both elements' faces toward the interface then lie on the same
  // side, so no consistent Upper/Lower labeling exists and the ElementId
  // tie-break decides. Use distinct block ids so the two elements are ordered.
  const ElementId<2> smaller_element{3, {{SegmentId{2, 1}, SegmentId{1, 0}}}};
  const ElementId<2> larger_element{7, {{SegmentId{2, 1}, SegmentId{1, 0}}}};
  REQUIRE(smaller_element < larger_element);

  // Reverses the xi axis: host upper_xi maps to neighbor lower_xi.
  const OrientationMap<2> reversing_xi{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::upper_eta()}}};
  // The mirrored call uses the neighbor's orientation, which is the inverse.
  const OrientationMap<2> reversing_xi_inverse = reversing_xi.inverse_map();

  // Confirm this really is the twisted case from both perspectives: each
  // element's neighbor face toward the interface lies on the same side it does.
  REQUIRE(reversing_xi(Direction<2>::upper_xi().opposite()).side() ==
          Side::Upper);
  REQUIRE(reversing_xi_inverse(Direction<2>::upper_xi().opposite()).side() ==
          Side::Upper);

  // smaller_element < larger_element, so the smaller element is the Upper
  // element of the twisted interface.
  CHECK(interface_orientation<2>(
            smaller_element, Direction<2>::upper_xi(),
            make_neighbors<2>(larger_element, reversing_xi),
            larger_element) == InterfaceOrientation::InteriorIsUpper);
  // From the larger element's perspective (its orientation toward the smaller
  // element is the inverse map): the larger element is the Lower element.
  CHECK(interface_orientation<2>(
            larger_element, Direction<2>::upper_xi(),
            make_neighbors<2>(smaller_element, reversing_xi_inverse),
            smaller_element) == InterfaceOrientation::InteriorIsLower);
}

void test_twisted_3d() {
  const ElementId<3> smaller_element{
      2, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}}};
  const ElementId<3> larger_element{
      5, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}}};
  REQUIRE(smaller_element < larger_element);

  // Reverses the zeta axis while permuting the other two.
  const OrientationMap<3> reversing_zeta{std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_xi(),
       Direction<3>::lower_zeta()}}};
  const OrientationMap<3> reversing_zeta_inverse = reversing_zeta.inverse_map();

  REQUIRE(reversing_zeta(Direction<3>::upper_zeta().opposite()).side() ==
          Side::Upper);
  REQUIRE(
      reversing_zeta_inverse(Direction<3>::upper_zeta().opposite()).side() ==
      Side::Upper);

  CHECK(interface_orientation<3>(
            smaller_element, Direction<3>::upper_zeta(),
            make_neighbors<3>(larger_element, reversing_zeta),
            larger_element) == InterfaceOrientation::InteriorIsUpper);
  CHECK(interface_orientation<3>(
            larger_element, Direction<3>::upper_zeta(),
            make_neighbors<3>(smaller_element, reversing_zeta_inverse),
            smaller_element) == InterfaceOrientation::InteriorIsLower);
}

void test_periodic_self_neighbor() {
  // A single periodic element is its own neighbor in both xi directions with a
  // trivial orientation. This is the aligned case (side_me != side_nb), so the
  // element is labeled purely by the side it looks toward, independent of the
  // fact that the neighbor id equals its own id.
  const ElementId<1> element{0, {{SegmentId{0, 0}}}};
  const auto aligned = OrientationMap<1>::create_aligned();

  CHECK(interface_orientation<1>(element, Direction<1>::upper_xi(),
                                 make_neighbors<1>(element, aligned),
                                 element) ==
        InterfaceOrientation::InteriorIsLower);
  CHECK(interface_orientation<1>(element, Direction<1>::lower_xi(),
                                 make_neighbors<1>(element, aligned),
                                 element) ==
        InterfaceOrientation::InteriorIsUpper);
}

void test_self_identification_error() {
  // A periodic self-neighbor whose orientation reverses the shared axis puts
  // both faces on the same side while the ids are equal: no antisymmetric
  // labeling exists, so this must ERROR.
  const ElementId<2> element{0, {{SegmentId{0, 0}, SegmentId{1, 0}}}};
  const OrientationMap<2> reversing_xi{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::upper_eta()}}};
  REQUIRE(reversing_xi(Direction<2>::upper_xi().opposite()).side() ==
          Side::Upper);

  CHECK_THROWS_WITH(
      interface_orientation<2>(element, Direction<2>::upper_xi(),
                               make_neighbors<2>(element, reversing_xi),
                               element),
      Catch::Matchers::ContainsSubstring("on a twisted interface"));
}

void test_host_labeled_mortar() {
  // A mortar with multiple non-conforming neighbors is labeled by the host
  // element's own id. All neighbor orientations agree that the neighbors sit
  // on the opposite side of the interface (as on the radial wedge-to-Ylm-shell
  // interface), so a unique label exists even though the passed neighbor id
  // equals the host id.
  const ElementId<3> element{
      12, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}};
  const auto aligned = OrientationMap<3>::create_aligned();
  // A quarter-turn about zeta: permutes xi/eta but keeps the zeta side.
  const OrientationMap<3> quarter_turn{std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}}};
  std::unordered_set<ElementId<3>> ids{};
  std::unordered_map<size_t, OrientationMap<3>> orientations{};
  for (size_t block = 0; block < 3; ++block) {
    ids.insert(ElementId<3>{
        block, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}});
    orientations[block] = block % 2 == 0 ? aligned : quarter_turn;
  }
  const Neighbors<3> neighbors{ids, orientations, false};

  CHECK(interface_orientation<3>(element, Direction<3>::lower_zeta(), neighbors,
                                 element) ==
        InterfaceOrientation::InteriorIsUpper);
  CHECK(interface_orientation<3>(element, Direction<3>::upper_zeta(), neighbors,
                                 element) ==
        InterfaceOrientation::InteriorIsLower);
}

void test_host_labeled_disagreement_error() {
  // Mixed neighbor orientations that place the neighbors on both sides of the
  // interface admit no single label and must ERROR.
  const ElementId<3> element{
      12, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}};
  const OrientationMap<3> zeta_reversing{std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
       Direction<3>::lower_zeta()}}};
  std::unordered_set<ElementId<3>> ids{};
  std::unordered_map<size_t, OrientationMap<3>> orientations{};
  ids.insert(
      ElementId<3>{0, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}});
  orientations[0] = OrientationMap<3>::create_aligned();
  ids.insert(
      ElementId<3>{1, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}});
  orientations[1] = zeta_reversing;
  const Neighbors<3> neighbors{ids, orientations, false};

  CHECK_THROWS_WITH(
      interface_orientation<3>(element, Direction<3>::lower_zeta(), neighbors,
                               element),
      Catch::Matchers::ContainsSubstring("orientations disagree"));
}

void test_stream_operator() {
  CHECK(get_output(InterfaceOrientation::InteriorIsUpper) == "InteriorIsUpper");
  CHECK(get_output(InterfaceOrientation::InteriorIsLower) == "InteriorIsLower");
  CHECK(get_output(InterfaceOrientation::ExternalBoundary) ==
        "ExternalBoundary");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.InterfaceOrientation",
                  "[Unit][Evolution]") {
  test_aligned_1d();
  test_aligned_2d();
  test_aligned_3d();
  test_twisted_2d();
  test_twisted_3d();
  test_periodic_self_neighbor();
  test_self_identification_error();
  test_host_labeled_mortar();
  test_host_labeled_disagreement_error();
  test_stream_operator();
}
}  // namespace evolution::dg
