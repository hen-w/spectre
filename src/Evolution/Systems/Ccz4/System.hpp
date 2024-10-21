// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the second-order CCZ4 system.
 */
namespace Ccz4 {
struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<
      Tags::ConformalMetric<DataVector, 3>, Tags::ATilde<DataVector, 3>,
      Tags::ConformalFactor<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>, Tags::Theta<DataVector>,
      Tags::GammaHat<DataVector, 3>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, 3>, Tags::Fieldb<DataVector, 3>,
      Tags::LogLapse<DataVector>, Tags::LogConformalFactor<DataVector>>>;

  using flux_variables = tmpl::list<>;

  using gradient_variables =
      tmpl::list<Tags::ConformalMetric<DataVector, 3>,
                 Tags::ATilde<DataVector, 3>, Tags::ConformalFactor<DataVector>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 Tags::Fieldb<DataVector, 3>, Tags::LogLapse<DataVector>,
                 Tags::LogConformalFactor<DataVector>>;

  using gradients_tags = gradient_variables;
};
}  // namespace Ccz4
