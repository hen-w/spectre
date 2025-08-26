// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4::fd {
/// Option tags for reconstruction
namespace OptionTags {
/// \brief Option tag for the reconstructor
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};
}  // namespace OptionTags

namespace Tags {
/// \brief Tag for the reconstructor
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<fd::Reconstructor>;
  using option_tags = tmpl::list<OptionTags::Reconstructor>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& reconstructor) {
    return reconstructor->get_clone();
  }
};

/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = tmpl::list<
    ::Ccz4::Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<DataVector, 3>, ::Ccz4::Tags::ConformalFactor<DataVector>,
    ::Ccz4::Tags::ATilde<DataVector, 3>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
    ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>;
}  // namespace Tags
}  // namespace Ccz4::fd
