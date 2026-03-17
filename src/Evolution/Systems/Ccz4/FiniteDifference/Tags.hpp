// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellSolver.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4::fd {
/// \brief Option tags for evolving SoCcz4 with finite difference
namespace OptionTags {
/// \brief Option tag for the reconstructor
struct Reconstructor {
  using type = std::unique_ptr<fd::Reconstructor>;

  static constexpr Options::String help = {"The reconstruction scheme to use."};
  using group = evolution::dg::subcell::OptionTags::SubcellSolverGroup;
};

/// \brief Option tag for whether to evolve the lapse and shift
struct EvolveLapseAndShift {
  using type = bool;

  static constexpr Options::String help = {
      "The option to use time-independent laspe and shift."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};

/// \brief Option tag for whether to use constrained evolution
///
/// When true, the determint of the conformal spatial metric is rescaled
/// to one and the trace of ATilde is removed using the rescaled metric
/// after every complete time step.
struct ConstrainedEvolution {
  using type = bool;

  static constexpr Options::String help = {
      "Whether to use constrained evolution."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};

/// \brief Option tag for the epsilon parameter of the Kreiss-Oliger dissipation
struct KreissOligerEpsilon {
  using type = double;

  static constexpr Options::String help = {
      "The epsilon parameter for Kreiss-Oliger dissipation."};
  using group = ::Ccz4::OptionTags::Ccz4Group;
};
}  // namespace OptionTags

/// \brief Tags for evolving SoCcz4 with finite difference
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

/*!
 * \brief Tag for whether to evolve the lapse and shif
 */
struct EvolveLapseAndShift : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::EvolveLapseAndShift>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool evolve_lapse_and_shift) {
    return evolve_lapse_and_shift;
  }
};

/*!
 * \brief Tag for whether to evolve the lapse and shift
 */
struct ConstrainedEvolution : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::ConstrainedEvolution>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool constrained_evolution) {
    return constrained_evolution;
  }
};

/*!
 * \brief Tag for the epsilon parameter of the Kreiss-Oliger dissipation
 */
struct KreissOligerEpsilon : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::KreissOligerEpsilon>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const double kreiss_oliger_epsilon) {
    return kreiss_oliger_epsilon;
  }
};

/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = System::variables_tag_list;

template <typename DataType>
struct ConstraintCharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataType, 3>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CVectorZero : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType>
struct CScalarPlus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CScalarMinus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ConstraintCharacteristicFields : db::SimpleTag {
  using type =
      Variables<tmpl::list<CVectorZero<DataType, Dim, Frame>,
                           CScalarPlus<DataType>, CScalarMinus<DataType>>>;
};

template <typename DataType>
struct RadiationCharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataType, 2>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CTensorPlus : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CTensorMinus : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct RadiationCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<CTensorPlus<DataType, Dim, Frame>,
                                    CTensorMinus<DataType, Dim, Frame>>>;
};

template <typename DataType>
struct CPhi : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CGamma : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CAlpha : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CK : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CTheta : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CBeta : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataType, 16>;
};

/// Individual Scalar tags for characteristic speeds (for observation).
/// See Ccz4::fd::characteristic_speeds() for the mapping of indices to
/// physical speeds.
template <typename DataType, size_t Index>
struct CharSpeed : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() {
    return "CharSpeed" + std::to_string(Index);
  }
};

/// Individual Scalar tags for constraint characteristic speeds.
/// Index 0: zero speed, 1: +alpha-beta^n, 2: -alpha-beta^n
template <typename DataType, size_t Index>
struct ConstraintCharSpeed : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() {
    return "ConstraintCharSpeed" + std::to_string(Index);
  }
};

/// Individual Scalar tags for radiation characteristic speeds.
/// Index 0: +alpha-beta^n, 1: -alpha-beta^n
template <typename DataType, size_t Index>
struct RadiationCharSpeed : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() {
    return "RadiationCharSpeed" + std::to_string(Index);
  }
};

template <typename DataType, size_t Dim, typename Frame>
struct UTensorPlus : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UTensorMinus : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UVector1Zero : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UVector2Plus : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UVector2Minus : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UVector3Plus : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct UVector3Minus : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType>
struct UScalar1Zero : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar2Plus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar2Minus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar3Plus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar3Minus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar4Plus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar4Minus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar5Plus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct UScalar5Minus : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<
      UTensorPlus<DataType, Dim, Frame>, UTensorMinus<DataType, Dim, Frame>,
      UVector1Zero<DataType, Dim, Frame>, UVector2Plus<DataType, Dim, Frame>,
      UVector2Minus<DataType, Dim, Frame>, UVector3Plus<DataType, Dim, Frame>,
      UVector3Minus<DataType, Dim, Frame>, UScalar1Zero<DataType>,
      UScalar2Plus<DataType>, UScalar2Minus<DataType>, UScalar3Plus<DataType>,
      UScalar3Minus<DataType>, UScalar4Plus<DataType>, UScalar4Minus<DataType>,
      UScalar5Plus<DataType>, UScalar5Minus<DataType>>>;
};

template <typename DataType, size_t Dim, typename Frame>
struct DnConformalMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType>
struct DnLapse : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct DnShift : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

template <typename DataType>
struct DnConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct EvolvedSpaceFromCharacteristicFields : db::SimpleTag {
  using type = Variables<
      tmpl::list<DnConformalMetric<DataType, Dim, Frame>, DnLapse<DataType>,
                 DnShift<DataType, Dim, Frame>, DnConformalFactor<DataType>,
                 ::Ccz4::Tags::ATilde<DataType, Dim, Frame>,
                 gr::Tags::TraceExtrinsicCurvature<DataType>,
                 ::Ccz4::Tags::Theta<DataType>,
                 ::Ccz4::Tags::GammaHat<DataType, Dim, Frame>,
                 ::Ccz4::Tags::AuxiliaryShiftB<DataType, Dim, Frame>>>;
};

/// Type aliases for the tag lists used by characteristic field Variables.
/// These are used with ::Tags::Variables<...> to create DataBox-compatible
/// parent tags whose subitems are the individual characteristic tensors.
template <typename DataType, size_t Dim, typename Frame>
using characteristic_fields_tags_list = tmpl::list<
    UTensorPlus<DataType, Dim, Frame>, UTensorMinus<DataType, Dim, Frame>,
    UVector1Zero<DataType, Dim, Frame>, UVector2Plus<DataType, Dim, Frame>,
    UVector2Minus<DataType, Dim, Frame>, UVector3Plus<DataType, Dim, Frame>,
    UVector3Minus<DataType, Dim, Frame>, UScalar1Zero<DataType>,
    UScalar2Plus<DataType>, UScalar2Minus<DataType>, UScalar3Plus<DataType>,
    UScalar3Minus<DataType>, UScalar4Plus<DataType>, UScalar4Minus<DataType>,
    UScalar5Plus<DataType>, UScalar5Minus<DataType>>;

template <typename DataType, size_t Dim, typename Frame>
using constraint_characteristic_fields_tags_list =
    tmpl::list<CVectorZero<DataType, Dim, Frame>, CScalarPlus<DataType>,
               CScalarMinus<DataType>>;

template <typename DataType, size_t Dim, typename Frame>
using radiation_characteristic_fields_tags_list =
    tmpl::list<CTensorPlus<DataType, Dim, Frame>,
               CTensorMinus<DataType, Dim, Frame>>;

/// Observer-friendly parent tags (derive from ::Tags::Variables so that
/// subitems machinery expands them in the DataBox).
template <size_t Dim, typename Frame>
using ObserverCharacteristicFieldsTag =
    ::Tags::Variables<characteristic_fields_tags_list<DataVector, Dim, Frame>>;

template <size_t Dim, typename Frame>
using ObserverConstraintCharacteristicFieldsTag = ::Tags::Variables<
    constraint_characteristic_fields_tags_list<DataVector, Dim, Frame>>;

template <size_t Dim, typename Frame>
using ObserverRadiationCharacteristicFieldsTag = ::Tags::Variables<
    radiation_characteristic_fields_tags_list<DataVector, Dim, Frame>>;

/// Tag lists for characteristic speeds (individual Scalar tags).
using characteristic_speeds_tags_list = tmpl::list<
    CharSpeed<DataVector, 0>, CharSpeed<DataVector, 1>,
    CharSpeed<DataVector, 2>, CharSpeed<DataVector, 3>,
    CharSpeed<DataVector, 4>, CharSpeed<DataVector, 5>,
    CharSpeed<DataVector, 6>, CharSpeed<DataVector, 7>,
    CharSpeed<DataVector, 8>, CharSpeed<DataVector, 9>,
    CharSpeed<DataVector, 10>, CharSpeed<DataVector, 11>,
    CharSpeed<DataVector, 12>, CharSpeed<DataVector, 13>,
    CharSpeed<DataVector, 14>, CharSpeed<DataVector, 15>>;

using constraint_characteristic_speeds_tags_list = tmpl::list<
    ConstraintCharSpeed<DataVector, 0>, ConstraintCharSpeed<DataVector, 1>,
    ConstraintCharSpeed<DataVector, 2>>;

using radiation_characteristic_speeds_tags_list = tmpl::list<
    RadiationCharSpeed<DataVector, 0>, RadiationCharSpeed<DataVector, 1>>;

using ObserverCharacteristicSpeedsTag =
    ::Tags::Variables<characteristic_speeds_tags_list>;

using ObserverConstraintCharacteristicSpeedsTag =
    ::Tags::Variables<constraint_characteristic_speeds_tags_list>;

using ObserverRadiationCharacteristicSpeedsTag =
    ::Tags::Variables<radiation_characteristic_speeds_tags_list>;

}  // namespace Tags
}  // namespace Ccz4::fd
