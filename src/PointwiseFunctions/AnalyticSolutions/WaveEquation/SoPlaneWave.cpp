// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SoPlaneWave.hpp"

#include <algorithm>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace SoScalarWave::Solutions {

template <size_t Dim>
SoPlaneWave<Dim>::SoPlaneWave(
    std::array<double, Dim> wave_vector, std::array<double, Dim> center,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> profile)
    : wave_vector_(std::move(wave_vector)),
      center_(std::move(center)),
      profile_(std::move(profile)),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
SoPlaneWave<Dim>::SoPlaneWave(const SoPlaneWave& other)
    : evolution::initial_data::InitialData(other),
      wave_vector_(other.wave_vector_),
      center_(other.center_),
      profile_(other.profile_->get_clone()),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
SoPlaneWave<Dim>& SoPlaneWave<Dim>::operator=(const SoPlaneWave& other) {
  wave_vector_ = other.wave_vector_;
  center_ = other.center_;
  omega_ = magnitude(wave_vector_);
  profile_ = other.profile_->get_clone();
  return *this;
}

template <size_t Dim>
std::unique_ptr<evolution::initial_data::InitialData>
SoPlaneWave<Dim>::get_clone() const {
  return std::make_unique<SoPlaneWave<Dim>>(*this);
}

template <size_t Dim>
SoPlaneWave<Dim>::SoPlaneWave(CkMigrateMessage* msg) : InitialData(msg) {}

template <size_t Dim>
template <typename T>
Scalar<T> SoPlaneWave<Dim>::psi(const tnsr::I<T, Dim>& x,
                                const double t) const {
  return Scalar<T>(profile_->operator()(u(x, t)));
}

template <size_t Dim>
template <typename T>
Scalar<T> SoPlaneWave<Dim>::dpsi_dt(const tnsr::I<T, Dim>& x,
                                    const double t) const {
  return Scalar<T>(-omega_ * profile_->first_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> SoPlaneWave<Dim>::dpsi_dx(const tnsr::I<T, Dim>& x,
                                          const double t) const {
  auto result = make_with_value<tnsr::i<T, Dim>>(x, 0.0);
  const auto du = profile_->first_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = gsl::at(wave_vector_, i) * du;
  }
  return result;
}

template <size_t Dim>
template <typename T>
Scalar<T> SoPlaneWave<Dim>::d2psi_dt2(const tnsr::I<T, Dim>& x,
                                      const double t) const {
  return Scalar<T>(square(omega_) * profile_->second_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> SoPlaneWave<Dim>::d2psi_dtdx(const tnsr::I<T, Dim>& x,
                                             const double t) const {
  auto result = make_with_value<tnsr::i<T, Dim>>(x, 0.0);
  const auto d2u = profile_->second_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = -omega_ * gsl::at(wave_vector_, i) * d2u;
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> SoPlaneWave<Dim>::d2psi_dxdx(const tnsr::I<T, Dim>& x,
                                              const double t) const {
  auto result = make_with_value<tnsr::ii<T, Dim>>(x, 0.0);
  const auto d2u = profile_->second_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result.get(i, j) =
          gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j) * d2u;
    }
  }
  return result;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>
SoPlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, double t,
    const tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> /*meta*/) const {
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> variables{
      psi(x, t), dpsi_dt(x, t), dpsi_dx(x, t)};
  get<Tags::Pi>(variables).get() *= -1.0;
  return variables;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>, Tags::BoundaryPsi>
SoPlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, const double t,
    const tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>,
                     Tags::BoundaryPsi> /*meta*/) const {
  auto base_vars =
      variables(x, t, tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>{});
  // BoundaryPsi initialized to Psi
  return {std::move(get<Tags::Psi>(base_vars)),
          std::move(get<Tags::Pi>(base_vars)),
          std::move(get<Tags::Phi<Dim>>(base_vars)), psi(x, t)};
}

template <size_t Dim>
tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                    ::Tags::dt<Tags::Phi<Dim>>>
SoPlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, double t,
    const tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                     ::Tags::dt<Tags::Phi<Dim>>> /*meta*/) const {
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim>>>
      dt_variables{dpsi_dt(x, t), d2psi_dt2(x, t), d2psi_dtdx(x, t)};
  get<::Tags::dt<Tags::Pi>>(dt_variables).get() *= -1.0;
  return dt_variables;
}

template <size_t Dim>
tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                    ::Tags::dt<Tags::Phi<Dim>>, ::Tags::dt<Tags::BoundaryPsi>>
SoPlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, const double t,
    const tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                     ::Tags::dt<Tags::Phi<Dim>>,
                     ::Tags::dt<Tags::BoundaryPsi>> /*meta*/) const {
  auto base_dt_vars =
      variables(x, t,
                tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                           ::Tags::dt<Tags::Phi<Dim>>>{});
  // dt<BoundaryPsi> = dpsi_dt (same as dt<Psi> = -Pi)
  return {std::move(get<::Tags::dt<Tags::Psi>>(base_dt_vars)),
          std::move(get<::Tags::dt<Tags::Pi>>(base_dt_vars)),
          std::move(get<::Tags::dt<Tags::Phi<Dim>>>(base_dt_vars)),
          dpsi_dt(x, t)};
}

template <size_t Dim>
void SoPlaneWave<Dim>::pup(PUP::er& p) {
  InitialData::pup(p);
  p | wave_vector_;
  p | center_;
  p | profile_;
  p | omega_;
}
template <size_t Dim>
bool operator==(const SoPlaneWave<Dim>& lhs, const SoPlaneWave<Dim>& rhs) {
  return (lhs.wave_vector_ == rhs.wave_vector_) and
         (lhs.center_ == rhs.center_) and
         (*(lhs.profile_) == *(rhs.profile_)) and (lhs.omega_ == rhs.omega_);
}

template <size_t Dim>
bool operator!=(const SoPlaneWave<Dim>& lhs, const SoPlaneWave<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
template <typename T>
T SoPlaneWave<Dim>::u(const tnsr::I<T, Dim>& x, const double t) const {
  auto result = make_with_value<T>(x, -omega_ * t);
  for (size_t d = 0; d < Dim; ++d) {
    result += gsl::at(wave_vector_, d) * (x.get(d) - gsl::at(center_, d));
  }
  return result;
}

template <size_t Dim>
PUP::able::PUP_ID SoPlaneWave<Dim>::my_PUP_ID = 0;
}  // namespace SoScalarWave::Solutions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template class SoScalarWave::Solutions::SoPlaneWave<DIM(data)>;  \
  template bool SoScalarWave::Solutions::operator==(               \
      const SoScalarWave::Solutions::SoPlaneWave<DIM(data)>& lhs,  \
      const SoScalarWave::Solutions::SoPlaneWave<DIM(data)>& rhs); \
  template bool SoScalarWave::Solutions::operator!=(               \
      const SoScalarWave::Solutions::SoPlaneWave<DIM(data)>& lhs,  \
      const SoScalarWave::Solutions::SoPlaneWave<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                               \
  template Scalar<DTYPE(data)>                                             \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::psi(                    \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const;           \
  template Scalar<DTYPE(data)>                                             \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::dpsi_dt(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const;           \
  template tnsr::i<DTYPE(data), DIM(data)>                                 \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::dpsi_dx(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;     \
  template Scalar<DTYPE(data)>                                             \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::d2psi_dt2(              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;     \
  template tnsr::i<DTYPE(data), DIM(data)>                                 \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::d2psi_dtdx(             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;     \
  template tnsr::ii<DTYPE(data), DIM(data)>                                \
  SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::d2psi_dxdx(             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;     \
  template DTYPE(data) SoScalarWave::Solutions::SoPlaneWave<DIM(data)>::u( \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
