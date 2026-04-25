// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4::fd {

/*!
 * \brief Degree-4 unlimited FD reconstruction for the CCZ4 system.
 *
 * Uses `::fd::reconstruction::detail::UnlimitedReconstructor<4>`, which
 * has a stencil width of 5 and therefore needs `ceil(5/2) = 3` ghost zones.
 */
class UnlimitedDeg4Prim : public Reconstructor {
 public:
  static constexpr size_t dim = 3;

  using volume_vars_tags = System::variables_tag_list;
  using recons_tags = tags_list_for_reconstruct;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Degree-4 unlimited reconstruction for CCZ4."};

  UnlimitedDeg4Prim() = default;
  UnlimitedDeg4Prim(UnlimitedDeg4Prim&&) = default;
  UnlimitedDeg4Prim& operator=(UnlimitedDeg4Prim&&) = default;
  UnlimitedDeg4Prim(const UnlimitedDeg4Prim&) = default;
  UnlimitedDeg4Prim& operator=(const UnlimitedDeg4Prim&) = default;
  ~UnlimitedDeg4Prim() override = default;

  explicit UnlimitedDeg4Prim(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(Reconstructor, UnlimitedDeg4Prim);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  static constexpr bool use_adaptive_order = false;

  void pup(PUP::er& p) override;

  size_t ghost_zone_size() const override { return 3; }

  using reconstruction_argument_tags =
      tmpl::list<::Tags::Variables<volume_vars_tags>,
                 domain::Tags::Element<dim>,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<dim>,
                 evolution::dg::subcell::Tags::Mesh<dim>>;

  void reconstruct(
      gsl::not_null<std::array<Variables<recons_tags>, dim>*>
          vars_on_lower_face,
      gsl::not_null<std::array<Variables<recons_tags>, dim>*>
          vars_on_upper_face,
      const Variables<volume_vars_tags>& volume_vars,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh) const;

  void reconstruct_fd_neighbor(
      gsl::not_null<Variables<recons_tags>*> vars_on_face,
      const Variables<volume_vars_tags>& subcell_volume_vars,
      const Element<dim>& element,
      const DirectionalIdMap<dim, evolution::dg::subcell::GhostData>&
          ghost_data,
      const Mesh<dim>& subcell_mesh,
      const Direction<dim> direction_to_reconstruct) const;
};

bool operator==(const UnlimitedDeg4Prim& /*lhs*/,
                const UnlimitedDeg4Prim& /*rhs*/);

bool operator!=(const UnlimitedDeg4Prim& lhs, const UnlimitedDeg4Prim& rhs);

}  // namespace Ccz4::fd
