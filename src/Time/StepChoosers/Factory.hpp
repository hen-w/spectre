// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ElementSizeCfl.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/LimitIncrease.hpp"
#include "Time/StepChoosers/Maximum.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace StepChoosers {
namespace Factory_detail {
// For a system with a list-valued `variables_tag` (split volume/boundary
// variables) the standard choosers operate on the first entry, which holds
// the volume variables. All entries share the same steps, so error control
// on additional entries requires system-specific choosers with distinct
// `ErrorControlSelector`s (see the Cce executables).
template <typename System>
using step_chooser_variables_tag =
    tmpl::conditional_t<tt::is_a_v<tmpl::list, typename System::variables_tag>,
                        tmpl::front<typename System::variables_tag>,
                        typename System::variables_tag>;

template <typename Use, typename System, bool HasCharSpeedFunctions>
using common_step_choosers = tmpl::push_back<
    tmpl::conditional_t<
        HasCharSpeedFunctions,
        tmpl::list<StepChoosers::Cfl<Frame::Inertial, System>,
                   StepChoosers::ElementSizeCfl<System::volume_dim, System>>,
        tmpl::list<>>,
    StepChoosers::Constant,
    StepChoosers::ErrorControl<Use, step_chooser_variables_tag<System>>,
    StepChoosers::LimitIncrease, StepChoosers::Maximum,
    StepChoosers::PreventRapidIncrease<step_chooser_variables_tag<System>>>;
}  // namespace Factory_detail

template <typename System, bool HasCharSpeedFunctions = true>
using standard_step_choosers =
    Factory_detail::common_step_choosers<StepChooserUse::LtsStep, System,
                                         HasCharSpeedFunctions>;

template <typename System, bool HasCharSpeedFunctions = true>
using standard_slab_choosers =
    tmpl::push_back<Factory_detail::common_step_choosers<
                        StepChooserUse::Slab, System, HasCharSpeedFunctions>,
                    StepChoosers::StepToTimes>;
}  // namespace StepChoosers
