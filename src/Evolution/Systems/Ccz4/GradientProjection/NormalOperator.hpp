// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection {
/// A vector \f$(w,r)\f$ of the normal-only subspace: 13 velocity components
/// followed by 10 potential-rate components.
using NormalSubspaceVector = std::array<double, number_of_normal_components>;

/*!
 * \brief The factor \f$F\f$ of the normal principal matrix
 * \f$A_n=JF\f$ of the temporary target \f$Y=(w,q_1,q_2,q_3)\f$.
 *
 * \details With the sign convention \f$Y_t=A^i\partial_iY+\ldots\f$ and
 * \f$A_n=n_iA^i\f$ for a covector \f$n_i\f$ (any normalization; the map is
 * linear in \f$n\f$), let \f$v=\beta^in_i\f$, \f$\nu^i=g^{ij}n_j\f$,
 * \f$\mathrm{tf}_gX=X-g\,\mathrm{tr}(g^{-1}X)/3\f$,
 * \f$N_{ij}=n_{(i}g_{j)k}\hat\Gamma^k\f$, \f$C_{ij}=n_{(i}q_{\ell,j)}\f$,
 * \f$L^\alpha_{ij}=n_{(i}q_{\alpha,j)}\f$ and
 * \f$U=\phi^2\mathrm{tf}_g[\alpha(N+C)-L^\alpha]\f$. Then
 * \f{align*}{
 * (A_nY)_a&=va-\alpha\phi^2\nu^iq_{H,i}+2\mathcal T_H^{-1}U,\\
 * (A_nY)_K&=vK+\alpha\phi^2(n_i\hat\Gamma^i+4\nu^iq_{\ell,i})
 *   -\phi^2\nu^iq_{\alpha,i},\\
 * (A_nY)_\Theta&=v\Theta+\alpha\phi^2(n_i\hat\Gamma^i/2+2\nu^iq_{\ell,i}),\\
 * (A_nY)_{\hat\Gamma}^k&=(A_nY)_b^k=v\hat\Gamma^k
 *   +\alpha\nu^k(2\Theta-4K/3)+\nu^iB_i{}^k+\nu^kB_i{}^i/3,\\
 * (A_nY)_{q_i}&=n_iR(u,w,q),
 * \f}
 * where \f$R\f$ is `potential_rhs` with the Gamma-driver parameter \f$f\f$
 * (\f$R_\beta=fb\f$). This function returns \f$FY=((A_nY)_w,R(u,w,q))\f$; the
 * full action is \f$A_nY=J(FY)\f$ with \f$J(w,r)=(w,n\otimes r)\f$. The
 * \f$\hat\Gamma\f$ and \f$b\f$ rows include the curl ordering of the volume
 * equations.
 */
NormalSubspaceVector normal_flux(const LogMetricChart& chart,
                                 const PotentialVector& u,
                                 const std::array<double, 3>& normal,
                                 const VelocityVector& w,
                                 const GradientVector& q, double f);

/// \f$J(w,r)=(w,n\otimes r)\f$, see `normal_flux`.
void embed_normal_subspace_vector(gsl::not_null<VelocityVector*> result_w,
                                  gsl::not_null<GradientVector*> result_q,
                                  const std::array<double, 3>& normal,
                                  const NormalSubspaceVector& z);

/// \f$A_nY\f$, see `normal_flux`.
void apply_normal_matrix(gsl::not_null<VelocityVector*> result_w,
                         gsl::not_null<GradientVector*> result_q,
                         const LogMetricChart& chart, const PotentialVector& u,
                         const std::array<double, 3>& normal,
                         const VelocityVector& w, const GradientVector& q,
                         double f);

/*!
 * \brief Analytic characteristic decomposition of the normal-only symbol
 * \f$M=FJ\f$ (\f$23\times23\f$), whose nonzero spectrum is the nonzero
 * spectrum of \f$A_n\f$.
 *
 * \details Since the \f$q\f$ rows of \f$A_n\f$ are proportional to \f$n\f$ and
 * the \f$b\f$ rows equal the \f$\hat\Gamma\f$ rows, \f$A_n=JF\f$ with
 * \f$M=FJ\f$ and the spectral absolute value is
 * \f$|A_n|=J\,\mathrm{sign}(M)\,F\f$, where \f$\mathrm{sign}(M)\f$ is
 * \f$\pm1\f$ on the eigenvectors of the twenty nonzero characteristic speeds
 * and \f$0\f$ on the three-dimensional structural kernel of \f$M\f$
 * (\f$\hat\Gamma-b\f$). This is \f$\sum_\lambda|\lambda|P_\lambda\f$ of
 * \f$A_n\f$ without dividing by any characteristic speed.
 *
 * The decomposition is constructed analytically:
 *
 * 1. With \f$X=e^H\f$ (\f$g=X^2\f$), the constant coordinate change
 *    \f$x'=Xx\f$ maps \f$g\f$ to the identity. The chart variables transform
 *    as \f$a'=\tfrac12X^{-1}(\mathcal T_Ha)X^{-1}\f$ (and likewise the
 *    \f$H\f$ rate), vectors as \f$v'=Xv\f$, scalars unchanged, and the normal
 *    covector as \f$n'=X^{-1}n\f$ with \f$m=|n'|=\sqrt{g^{ij}n_in_j}\f$.
 *    Because \f$M\f$ is quadratic in the normal through
 *    \f$q=n\otimes r\f$, \f$M(n')=D(mM(\hat n))D^{-1}\f$ with
 *    \f$D=\mathrm{diag}(1_w,m^{-1}1_r)\f$ and \f$\hat n=n'/m\f$, and the
 *    advection speed becomes \f$\hat v=\beta^in_i/m\f$.
 * 2. In this isotropic frame the symbol splits, relative to \f$\hat n\f$ and a
 *    fixed orthonormal tangent basis, into two transverse-traceless pairs,
 *    two five-component vector sectors and one nine-component scalar sector.
 * 3. Explicit triangular changes of variables decouple each sector into
 *    \f$2\times2\f$ blocks \f$\begin{pmatrix}\hat v&p\\q&\hat v\end{pmatrix}\f$
 *    and Gamma-driver shift blocks: in the vector sector
 *    \f$\tilde y=y-\hat\Gamma_t/2\f$; in the scalar sector
 *    \f$\tilde x=x-(K-2\Theta)\f$, \f$\tilde y=y-(\hat\Gamma_n+\ell)\f$,
 *    \f$P=K-2\Theta\f$, \f$\psi=\hat\Gamma_n+4\ell\f$, and the longitudinal
 *    shift block is decoupled from \f$(\Theta,\psi)\f$ and \f$(P,s)\f$ by
 *    explicit Sylvester solutions.
 *
 * With the Gamma-driver parameter \f$f>0\f$ (\f$\partial_t\beta^i=fb^i\f$),
 * each shift block acts on the transverse or longitudinal components of
 * \f$(\hat\Gamma,b,\beta)\f$ as \f$\hat\Gamma'=b'=\hat v\hat\Gamma+c_s\beta\f$,
 * \f$\beta'=fb\f$, with \f$c_s=1\f$ (transverse) or \f$c_s=4/3\f$
 * (longitudinal). Its kernel field is \f$\hat\Gamma-b\f$, and on
 * \f$(b,(\beta+\hat v(\hat\Gamma-b)/c_s)/f)\f$ it reduces to
 * \f$\begin{pmatrix}\hat v&c_sf\\1&0\end{pmatrix}\f$. The changes of variables
 * of the light-cone, \f$\Theta\f$ and 1+log blocks do not depend on \f$f\f$;
 * the Sylvester couplings do.
 *
 * The characteristic speeds are, with \f$v=\beta^in_i\f$,
 * \f$v\pm\alpha\phi m\f$ (light cone and \f$\Theta\f$, multiplicity 6 each),
 * \f$v\pm\phi m\sqrt{2\alpha}\f$ (1+log),
 * \f$v/2\pm\sqrt{v^2/4+fm^2}\f$ (transverse shift, multiplicity 2),
 * \f$v/2\pm\sqrt{v^2/4+4fm^2/3}\f$ (longitudinal shift) and three structural
 * zeros. For a physical-unit normal \f$n/s\f$ with
 * \f$s=\phi m\f$ all speeds and the absolute matrix scale by \f$1/s\f$.
 *
 * It is an `ERROR` if \f$f\f$ is not positive, if a nonzero characteristic
 * speed vanishes to within a relative tolerance (a sonic point, where
 * \f$A_n\f$ is not diagonalizable) or if the longitudinal shift speed collides
 * with a light-cone or 1+log speed (where the puncture gauge loses strong
 * hyperbolicity; in Minkowski space with unit lapse and zero shift this
 * happens at \f$f=3/4\f$ and \f$f=3/2\f$). The degeneracies are reported with
 * the state; none is regularized.
 *
 * The order of the 23 characteristic fields is: transverse-traceless
 * \f$(+,-)\f$ for the two polarizations; for each of the two tangent
 * directions, light \f$(+,-)\f$, shift \f$(+,-)\f$ and the shift kernel;
 * scalar light \f$(+,-)\f$, \f$\Theta\f$ \f$(+,-)\f$, 1+log \f$(+,-)\f$,
 * longitudinal shift \f$(+,-)\f$ and the scalar shift kernel.
 */
class NormalCharacteristics {
 public:
  NormalCharacteristics(const LogMetricChart& chart, const PotentialVector& u,
                        const std::array<double, 3>& normal, double f);

  /// The eigenvalues of \f$M\f$, ordered as the characteristic fields.
  const NormalSubspaceVector& speeds() const { return speeds_; }

  /// Characteristic fields \f$c=L z\f$ of a normal-subspace vector.
  NormalSubspaceVector characteristic_fields(
      const NormalSubspaceVector& z) const;

  /// The inverse transform \f$z=R c\f$ of `characteristic_fields`.
  NormalSubspaceVector normal_fields_from_characteristic_fields(
      const NormalSubspaceVector& c) const;

  /// \f$\mathrm{sign}(M)z\f$, zero on the structural kernel.
  NormalSubspaceVector apply_sign(const NormalSubspaceVector& z) const;

  /// Whether the given characteristic field is in the structural kernel.
  static bool is_kernel_field(size_t index);

 private:
  NormalSubspaceVector to_isotropic(const NormalSubspaceVector& z) const;
  NormalSubspaceVector from_isotropic(const NormalSubspaceVector& z) const;

  LogMetricChart chart_;
  Matrix3 sqrt_metric_{};
  Matrix3 inverse_sqrt_metric_{};
  std::array<double, 3> unit_normal_{};
  std::array<double, 3> tangent_1_{};
  std::array<double, 3> tangent_2_{};
  double normal_magnitude_ = 0.0;
  double advection_ = 0.0;
  double lapse_ = 0.0;
  double conformal_factor_ = 0.0;
  double f_ = 0.0;
  // Decoupling of the longitudinal shift block from (Theta, psi) and (P, s).
  std::array<double, 4> theta_coupling_{};
  std::array<double, 4> lapse_coupling_{};
  NormalSubspaceVector speeds_{};
};

/*!
 * \brief The spectral absolute value \f$|A_n|Y=J\,\mathrm{sign}(M)FY\f$, see
 * `NormalCharacteristics`.
 */
void apply_absolute_normal_matrix(gsl::not_null<VelocityVector*> result_w,
                                  gsl::not_null<GradientVector*> result_q,
                                  const LogMetricChart& chart,
                                  const PotentialVector& u,
                                  const std::array<double, 3>& normal,
                                  const VelocityVector& w,
                                  const GradientVector& q, double f);

/// The largest characteristic speed magnitude of \f$A_n\f$, evaluated from
/// the analytic speeds without the degeneracy checks of
/// `NormalCharacteristics`. It is an `ERROR` if \f$f\f$ is not positive.
double largest_characteristic_speed(const LogMetricChart& chart,
                                    const PotentialVector& u,
                                    const std::array<double, 3>& normal,
                                    double f);
}  // namespace Ccz4::GradientProjection
