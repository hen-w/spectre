// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace PUP {
class er;
}  // namespace PUP

/// \endcond

namespace intrp {

/// \ingroup NumericalAlgorithmsGroup
/// \brief Interpolates a `Variables` onto an arbitrary set of points.
///
/// \details If the `source_mesh` uses Spectral::Basis::FiniteDifference, each
/// target point is interpolated with a tensor product of 1D Lagrange stencils
/// of degree `fd_to_fd_interp_order` (1 to 6, linear if not specified) on
/// consecutive source points, which are (nearly) centered on the target point
/// in the interior of the source grid and one-sided near and beyond its ends.
/// If all dimensions are finite difference, only the weights of each target
/// point's stencil are stored and applied; a source mesh combining finite
/// difference and Cartoon bases uses a dense interpolation matrix. Otherwise
/// it uses the barycentric interpolation provided by
/// Spectral::interpolation_matrix in each dimension.
template <size_t Dim>
class Irregular {
 public:
  Irregular(
      const Mesh<Dim>& source_mesh,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>& target_points,
      const std::optional<size_t>& fd_to_fd_interp_order = std::nullopt);
  Irregular(const Mesh<Dim>& source_mesh,
            const tnsr::I<double, Dim, Frame::ElementLogical>& target_point,
            const std::optional<size_t>& fd_to_fd_interp_order = std::nullopt);
  Irregular();

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// @{
  /// Performs the interpolation on a `Variables` with grid points corresponding
  /// to the `Mesh<Dim>` specified in the constructor.
  /// The result is a `Variables` whose internal `DataVector` goes over the
  /// list of target_points that were specified in the constructor.
  /// \note for the void function, `result` will be resized to the proper size.
  template <typename TagsList>
  void interpolate(gsl::not_null<Variables<TagsList>*> result,
                   const Variables<TagsList>& vars) const;
  template <typename TagsList>
  Variables<TagsList> interpolate(const Variables<TagsList>& vars) const;
  /// @}

  /// @{
  /// \brief Interpolate a DataVector or ComplexDataVector onto the target
  /// points.
  ///
  /// \note When interpolating multiple tensors, the Variables interface is more
  /// efficient. However, this DataVector interface is useful for applications
  /// where only some components of a Tensor or Variables need to be
  /// interpolated.
  void interpolate(gsl::not_null<DataVector*> result,
                   const DataVector& input) const;
  DataVector interpolate(const DataVector& input) const;
  void interpolate(gsl::not_null<ComplexDataVector*> result,
                   const ComplexDataVector& input) const;
  ComplexDataVector interpolate(const ComplexDataVector& input) const;
  /// @}

  /// \brief Interpolate multiple variables on the grid to the target points.
  /// @{
  void interpolate(gsl::not_null<gsl::span<double>*> result,
                   const gsl::span<const double>& input) const;
  void interpolate(gsl::not_null<gsl::span<std::complex<double>>*> result,
                   const gsl::span<const std::complex<double>>& input) const;
  void interpolate(gsl::not_null<gsl::span<float>*> result,
                   const gsl::span<const float>& input) const;
  /// @}

 private:
  friend bool operator==(const Irregular& lhs, const Irregular& rhs) {
    return lhs.uses_fd_stencils_ == rhs.uses_fd_stencils_ and
           lhs.interpolation_matrix_ == rhs.interpolation_matrix_ and
           lhs.number_of_source_points_ == rhs.number_of_source_points_ and
           lhs.fd_window_starts_ == rhs.fd_window_starts_ and
           lhs.fd_stencil_offsets_ == rhs.fd_stencil_offsets_ and
           lhs.fd_weights_ == rhs.fd_weights_;
  }

  template <typename DataType>
  void initialize(
      const Mesh<Dim>& source_mesh,
      const tnsr::I<DataType, Dim, Frame::ElementLogical>& target_points,
      const std::optional<size_t>& fd_to_fd_interp_order);

  template <typename ValueType>
  void interpolate_with_fd_stencils(gsl::span<ValueType> result,
                                    gsl::span<const ValueType> input) const;

  size_t number_of_target_points() const {
    return uses_fd_stencils_ ? fd_window_starts_.size()
                             : interpolation_matrix_.rows();
  }
  size_t number_of_source_points() const {
    return uses_fd_stencils_ ? number_of_source_points_
                             : interpolation_matrix_.columns();
  }

  // Whether the source mesh is finite difference in all dimensions, in which
  // case the `fd_` members below are used instead of `interpolation_matrix_`.
  bool uses_fd_stencils_{false};
  // Dense (number of target points) x (number of source points) matrix
  Matrix interpolation_matrix_{};
  size_t number_of_source_points_{0};
  // The interpolated value at target point `p` is the sum over `q` of
  // `fd_weights_[p * fd_stencil_offsets_.size() + q]` times the source value
  // at `fd_window_starts_[p] + fd_stencil_offsets_[q]`. The offsets enumerate
  // the tensor-product stencil window with the first dimension varying fastest,
  // like the source points.
  std::vector<size_t> fd_window_starts_{};
  std::vector<size_t> fd_stencil_offsets_{};
  std::vector<double> fd_weights_{};
};

template <size_t Dim>
template <typename TagsList>
void Irregular<Dim>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const {
  if (UNLIKELY(result->number_of_grid_points() != number_of_target_points())) {
    *result = Variables<TagsList>(number_of_target_points(), 0.);
  }
  ASSERT(number_of_source_points() == vars.number_of_grid_points(),
         "Number of grid points in source 'vars', "
             << vars.number_of_grid_points()
             << ",\n disagrees with the size of the source_mesh, "
             << number_of_source_points()
             << ", that was passed into the constructor");
  auto result_span = gsl::make_span(result->data(), result->size());
  const auto vars_span = gsl::make_span(vars.data(), vars.size());
  interpolate(make_not_null(&result_span), vars_span);
}

template <size_t Dim>
template <typename TagsList>
Variables<TagsList> Irregular<Dim>::interpolate(
    const Variables<TagsList>& vars) const {
  Variables<TagsList> result{number_of_target_points()};
  interpolate(make_not_null(&result), vars);
  return result;
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs, const Irregular<Dim>& rhs);

}  // namespace intrp
