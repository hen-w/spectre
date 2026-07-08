// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace SoScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
struct BoundaryPsi;
}  // namespace SoScalarWave::Tags
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SoScalarWave::Solutions {
/*!
 * \brief A standing wave solution to the Euclidean wave equation
 *
 * The solution is given by
 * \f$\Psi(\vec{x},t) = A \sin(\vec{k} \cdot (\vec{x} - \vec{x_0}))
 * \cos(\omega t)\f$
 * with the wave vector \f$\vec{k}\f$, frequency
 * \f$\omega = ||\vec{k}||\f$, amplitude \f$A\f$, and center
 * \f$\vec{x_0}\f$.
 *
 * At \f$t = 0\f$ this gives \f$\Pi = 0\f$, meaning the initial data
 * decomposes into equal left-moving and right-moving components.
 *
 * \tparam Dim the spatial dimension of the solution
 */
template <size_t Dim>
class SoStandingWave : public evolution::initial_data::InitialData,
                       public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = Dim;

  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The wave vector of the standing wave."};
  };

  struct Center {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The center of the spatial profile."};
  };

  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the standing wave."};
  };

  using options = tmpl::list<WaveVector, Center, Amplitude>;

  static constexpr Options::String help = {
      "A standing wave solution of the Euclidean wave equation. "
      "Psi = A sin(k.(x-x0)) cos(omega t), with omega = |k|. "
      "At t=0, Pi=0 so the wave has equal left- and right-moving components."};

  using tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi,
                 ::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                 ::Tags::dt<Tags::Phi<Dim>>, ::Tags::dt<Tags::BoundaryPsi>>;

  SoStandingWave() = default;
  SoStandingWave(std::array<double, Dim> wave_vector,
                 std::array<double, Dim> center, double amplitude);
  SoStandingWave(const SoStandingWave&) = default;
  SoStandingWave& operator=(const SoStandingWave&) = default;
  SoStandingWave(SoStandingWave&&) = default;
  SoStandingWave& operator=(SoStandingWave&&) = default;
  ~SoStandingWave() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit SoStandingWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SoStandingWave);
  /// \endcond

  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> /*meta*/) const;

  /// Retrieve the evolution variables including BoundaryPsi
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi>
            /*meta*/) const;

  /// Retrieve the LDG evolved variables (Psi, Pi, BoundaryPsi); Phi is an
  /// auxiliary variable and is not part of `variables_tag`.
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::BoundaryPsi> variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<Tags::Psi, Tags::Pi, Tags::BoundaryPsi> /*meta*/) const;

  /// Retrieve the time derivatives of the evolution variables
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim>>>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                       ::Tags::dt<Tags::Phi<Dim>>> /*meta*/) const;

  /// Retrieve time derivatives including dt<BoundaryPsi>
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim>>, ::Tags::dt<Tags::BoundaryPsi>>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                       ::Tags::dt<Tags::Phi<Dim>>,
                       ::Tags::dt<Tags::BoundaryPsi>> /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const SoStandingWave<LocalDim>& lhs,
                         const SoStandingWave<LocalDim>& rhs);
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator!=(const SoStandingWave<LocalDim>& lhs,
                         const SoStandingWave<LocalDim>& rhs);

  std::array<double, Dim> wave_vector_{};
  std::array<double, Dim> center_{};
  double amplitude_{};
  double omega_{};
};
}  // namespace SoScalarWave::Solutions
