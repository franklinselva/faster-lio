#ifndef FASTER_LIO_WHEEL_FUSION_H
#define FASTER_LIO_WHEEL_FUSION_H

#include <Eigen/Core>
#include <cmath>

#include "faster_lio/use-ikfom.hpp"

namespace faster_lio::wheel_fusion {

/// State tangent-space dimension for the Faster-LIO state_ikfom manifold.
/// Layout: pos 0..2, rot 3..5, offset_R 6..8, offset_T 9..11, vel 12..14,
///         bg 15..17, ba 18..20, grav 21..22.
constexpr int kStateDof = 23;
using StateCov = Eigen::Matrix<double, kStateDof, kStateDof>;

/// Outcome of a scalar update attempt. `updated` tells the caller whether
/// (x, P) were mutated. `mahalanobis2 = y²/S` (squared Mahalanobis distance
/// of the innovation, 1-DOF). `residual = z − h(x)`.
struct ScalarUpdateReport {
    bool   updated       = false;
    double residual      = 0.0;
    double innov_cov     = 0.0;
    double mahalanobis2  = 0.0;
    enum class Status {
        Applied,
        InvalidAxis,
        NonPositiveR,
        NonFiniteInnov,
        GatedByMahalanobis,
    } status = Status::Applied;
};

/// Apply one scalar body-frame velocity Kalman update to (x, P).
///   z = (R_wb^T · v_world)[axis]
/// Jacobian rows (nonzero only in rot δθ 3..5 and vel 12..14 blocks):
///   ∂z/∂δθ       = [v_body]×[axis, :]
///   ∂z/∂v_world  = R_wb^T  [axis, :]
/// Joseph-form covariance update + symmetrization for numerical stability.
/// Mahalanobis gate rejects clearly divergent single samples (default 7σ²).
inline ScalarUpdateReport ApplyScalarBodyVelUpdate(
    state_ikfom &x, StateCov &P,
    int axis, double z_body, double R_obs,
    double mahalanobis_gate_sq = 50.0) {
    ScalarUpdateReport rep{};

    if (axis < 0 || axis > 2) {
        rep.status = ScalarUpdateReport::Status::InvalidAxis;
        return rep;
    }
    if (!(R_obs > 0.0) || !std::isfinite(R_obs)) {
        rep.status = ScalarUpdateReport::Status::NonPositiveR;
        return rep;
    }
    if (!std::isfinite(z_body)) {
        // MTK's boxplus has a debug assert on NaN. Reject upstream so a
        // stray NaN from a bad wheel sample does not abort the process.
        rep.status = ScalarUpdateReport::Status::NonFiniteInnov;
        return rep;
    }

    const Eigen::Matrix3d R_wb = x.rot.toRotationMatrix();
    const Eigen::Vector3d v_world(x.vel[0], x.vel[1], x.vel[2]);
    const Eigen::Vector3d v_body = R_wb.transpose() * v_world;
    const double z_pred = v_body[axis];
    rep.residual = z_body - z_pred;

    // Build 1×DOF Jacobian H.
    Eigen::Matrix<double, 1, kStateDof> H = Eigen::Matrix<double, 1, kStateDof>::Zero();
    Eigen::Matrix3d v_body_skew;
    v_body_skew <<         0.0, -v_body.z(),  v_body.y(),
                   v_body.z(),         0.0, -v_body.x(),
                  -v_body.y(),  v_body.x(),         0.0;
    H.block<1, 3>(0, 3)  = v_body_skew.row(axis);   // ∂/∂δθ
    H.block<1, 3>(0, 12) = R_wb.transpose().row(axis);  // ∂/∂v_world

    Eigen::Matrix<double, kStateDof, 1> PHt = P * H.transpose();
    const double S = (H * PHt)(0, 0) + R_obs;
    rep.innov_cov = S;
    if (!(S > 0.0) || !std::isfinite(S)) {
        rep.status = ScalarUpdateReport::Status::NonFiniteInnov;
        return rep;
    }
    rep.mahalanobis2 = (rep.residual * rep.residual) / S;
    if (rep.mahalanobis2 > mahalanobis_gate_sq) {
        rep.status = ScalarUpdateReport::Status::GatedByMahalanobis;
        return rep;
    }

    Eigen::Matrix<double, kStateDof, 1> K = PHt / S;
    Eigen::Matrix<double, kStateDof, 1> dx = K * rep.residual;

    x.boxplus(dx);  // manifold-aware; handles SO(3)/S2 internally

    // Joseph-form covariance update + symmetrize.
    Eigen::Matrix<double, kStateDof, kStateDof> I_KH =
        Eigen::Matrix<double, kStateDof, kStateDof>::Identity() - K * H;
    StateCov P_new = I_KH * P * I_KH.transpose() + K * R_obs * K.transpose();
    P = 0.5 * (P_new + P_new.transpose());

    rep.updated = true;
    rep.status = ScalarUpdateReport::Status::Applied;
    return rep;
}

}  // namespace faster_lio::wheel_fusion

#endif  // FASTER_LIO_WHEEL_FUSION_H
