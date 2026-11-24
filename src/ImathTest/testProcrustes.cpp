//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <ImathEuler.h>
#include <ImathMatrixAlgo.h>
#include <ImathRandom.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// Verify that if our transformation is already orthogonal, procrustes doesn't
// change that:
template <typename T>
void
testTranslationRotationMatrix (const IMATH_NAMESPACE::M44d& mat)
{
    std::cout << "Testing known translate/rotate matrix:\n " << mat;
    typedef IMATH_NAMESPACE::Vec3<T> Vec;

    static IMATH_NAMESPACE::Rand48 rand (2047);

    size_t           numPoints = 7;
    std::vector<Vec> from;
    from.reserve (numPoints);
    std::vector<Vec> to;
    to.reserve (numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        IMATH_NAMESPACE::V3d a (
            rand.nextf (), rand.nextf (), rand.nextf ());
        IMATH_NAMESPACE::V3d b = a * mat;

        from.push_back (Vec (a));
        to.push_back (Vec (b));
    }

    std::vector<T>                       weights (numPoints, T (1));
    const IMATH_NAMESPACE::M44d m1 = procrustesRotationAndTranslation (
        &from[0], &to[0], &weights[0], numPoints);
    const IMATH_NAMESPACE::M44d m2 =
        procrustesRotationAndTranslation (&from[0], &to[0], numPoints);

    const T eps = sizeof (T) == 8 ? T(1e-8) : T(1e-4);
    for (size_t i = 0; i < numPoints; ++i)
    {
        const IMATH_NAMESPACE::V3d a  = from[i];
        const IMATH_NAMESPACE::V3d b  = to[i];
        const IMATH_NAMESPACE::V3d b1 = a * m1;
        const IMATH_NAMESPACE::V3d b2 = a * m2;

        assert ((b - b1).length () < eps);
        assert ((b - b2).length () < eps);
    }
    std::cout << "  OK\n";
}

// Test that if we pass in a matrix that we know consists only of translates,
// rotates, and uniform scale that we get an exact match.
template <typename T>
void
testWithTranslateRotateAndScale (const IMATH_NAMESPACE::M44d& m)
{
    std::cout << "Testing with known translate/rotate/scale matrix\n" << m;
    IMATH_NAMESPACE::Rand48 rand (5376);

    typedef IMATH_NAMESPACE::Vec3<T> V3;
    std::vector<V3>                           from;
    std::vector<T>                            weights;

    const float eps = 1e-4f;
    std::cout << "numPoints: " << std::flush;
    for (size_t numPoints = 1; numPoints < 10; ++numPoints)
    {
        from.push_back (V3(
            static_cast<T>(rand.nextf()),
            static_cast<T>(rand.nextf()),
            static_cast<T>(rand.nextf())));
        weights.push_back(static_cast<T>(rand.nextf()));
        std::cout << from.size () << " ";

        std::vector<V3> to;
        for (size_t i = 0; i < from.size (); ++i)
            to.push_back (from[i] * m);

        // weighted:
        IMATH_NAMESPACE::M44d res =
            IMATH_NAMESPACE::procrustesRotationAndTranslation (
                &from[0], &to[0], &weights[0], from.size (), true);
        for (size_t i = 0; i < from.size (); ++i)
            assert ((from[i] * res - to[i]).length () < eps);

        // unweighted:
        res = IMATH_NAMESPACE::procrustesRotationAndTranslation (
            &from[0], &to[0], from.size (), true);
        for (size_t i = 0; i < from.size (); ++i)
            assert ((from[i] * res - to[i]).length () < eps);
    }
    std::cout << "  OK\n";
}

template <typename T>
double
procrustesError (
    const IMATH_NAMESPACE::Vec3<T>* from,
    const IMATH_NAMESPACE::Vec3<T>* to,
    const T*                                 weights,
    const size_t                             n,
    const IMATH_NAMESPACE::M44d&    xform)
{
    double result   = 0.0;
    double residual = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        IMATH_NAMESPACE::V3d xformed =
            IMATH_NAMESPACE::V3d (from[i]) * xform;
        IMATH_NAMESPACE::V3d diff =
            xformed - IMATH_NAMESPACE::V3d (to[i]);
        const double w   = weights[i];
        const double mag = w * diff.length2 ();

        // Use Kahan summation for the heck of it:
        const double y = mag - residual;
        const double t = result + y;
        residual       = (t - result) - y;
        result         = t;
    }
    return result;
}

template <typename T>
void
verifyProcrustes (
    const std::vector<IMATH_NAMESPACE::Vec3<T>>& from,
    const std::vector<IMATH_NAMESPACE::Vec3<T>>& to)
{
    const T eps = std::sqrt (std::numeric_limits<T>::epsilon ());

    const size_t n = from.size ();

    // Validate that passing in uniform weights gives the same answer as
    // passing in no weights:
    std::vector<T> weights (from.size ());
    for (size_t i = 0; i < weights.size (); ++i)
        weights[i] = 1;
    IMATH_NAMESPACE::M44d m1 =
        procrustesRotationAndTranslation (&from[0], &to[0], n);
    IMATH_NAMESPACE::M44d m2 =
        procrustesRotationAndTranslation (&from[0], &to[0], &weights[0], n);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            assert (std::abs (m1[i][j] - m2[i][j]) < eps);

    // Now try the weighted version:
    for (size_t i = 0; i < weights.size (); ++i)
        weights[i] = static_cast<T>(i + 1);

    IMATH_NAMESPACE::M44d m =
        procrustesRotationAndTranslation (&from[0], &to[0], &weights[0], n);

    // with scale:
    IMATH_NAMESPACE::M44d ms = procrustesRotationAndTranslation (
        &from[0], &to[0], &weights[0], n, true);

    // Verify that it's orthonormal w/ positive determinant.
    const T det = static_cast<T>(m.determinant ());
    assert (std::abs (det - T (1)) < eps);

    // Verify orthonormal:
    IMATH_NAMESPACE::M33d upperLeft;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            upperLeft[i][j] = m[i][j];
    IMATH_NAMESPACE::M33d product =
        upperLeft * upperLeft.transposed ();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double expected = (i == j ? 1.0 : 0.0);
            assert (std::abs (product[i][j] - expected) < eps);
        }
    }

    // Verify that nearby transforms are worse:
    const size_t                     numTries = 10;
    IMATH_NAMESPACE::Rand48 rand (1056);
    const double                     delta = 1e-3;
    for (size_t i = 0; i < numTries; ++i)
    {
        // Construct an orthogonal rotation matrix using Euler angles:
        IMATH_NAMESPACE::Eulerd diffRot (
            delta * rand.nextf (),
            delta * rand.nextf (),
            delta * rand.nextf ());

        assert (
            procrustesError (
                &from[0], &to[0], &weights[0], n, m * diffRot.toMatrix44 ()) >
            procrustesError (&from[0], &to[0], &weights[0], n, m));

        // Try a small translation:
        IMATH_NAMESPACE::V3d diffTrans (
            delta * rand.nextf (),
            delta * rand.nextf (),
            delta * rand.nextf ());
        IMATH_NAMESPACE::M44d translateMatrix;
        translateMatrix.translate (diffTrans);
        assert (
            procrustesError (
                &from[0], &to[0], &weights[0], n, m * translateMatrix) >
            procrustesError (&from[0], &to[0], &weights[0], n, m));
    }

    // Try a small scale:
    IMATH_NAMESPACE::M44d newMat    = ms;
    const double                   scaleDiff = delta;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            newMat[i][j] = ms[i][j] * (1.0 + scaleDiff);
    assert (
        procrustesError (&from[0], &to[0], &weights[0], n, newMat) >
        procrustesError (&from[0], &to[0], &weights[0], n, ms));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            newMat[i][j] = ms[i][j] * (1.0 - scaleDiff);
    assert (
        procrustesError (&from[0], &to[0], &weights[0], n, newMat) >
        procrustesError (&from[0], &to[0], &weights[0], n, ms));

    //
    // Verify the magical property that makes shape springs work:
    // when the displacements Q*A-B, times the weights,
    // are applied as forces at B,
    // there is zero net force and zero net torque.
    //
    {
        IMATH_NAMESPACE::V3d netForce (0);
        IMATH_NAMESPACE::V3d netTorque (0);
        for (size_t iPoint = 0; iPoint < n; ++iPoint)
        {
            const IMATH_NAMESPACE::V3d force =
                weights[iPoint] * (from[iPoint] * m - to[iPoint]);
            netForce += force;
            netTorque += to[iPoint].cross (force);
        }

        assert (netForce.length2 () < eps);
        assert (netTorque.length2 () < eps);
    }
}

template <typename T>
void
testProcrustesWithMatrix (const IMATH_NAMESPACE::M44d& m)
{
    std::cout << "Testing Procrustes algorithm with arbitrary matrix: \n" << m;
    std::vector<IMATH_NAMESPACE::Vec3<T>> fromPoints;
    std::vector<IMATH_NAMESPACE::Vec3<T>> toPoints;

    IMATH_NAMESPACE::Rand48 random (1209);
    std::cout << "   numPoints: ";
    for (size_t numPoints = 1; numPoints < 10; ++numPoints)
    {
        std::cout << numPoints << " " << std::flush;
        fromPoints.clear ();
        toPoints.clear ();
        for (size_t i = 0; i < numPoints; ++i)
        {
            const IMATH_NAMESPACE::V3d fromPt (
                random.nextf (), random.nextf (), random.nextf ());
            const IMATH_NAMESPACE::V3d toPt = fromPt * m;
            fromPoints.push_back (IMATH_NAMESPACE::Vec3<T> (fromPt));
            toPoints.push_back (IMATH_NAMESPACE::Vec3<T> (toPt));
        }
        verifyProcrustes (fromPoints, toPoints);
    }
    std::cout << "OK\n";
}

// ----------------------------------------------------------------------------
// Scenario tests: corresponding markers from two scans of the same prop.
//
// Every marker has a measurement confidence, used as its weight.  The
// registration maps the first scan onto the second and minimizes the
// weighted sum of squared correspondence distances:
//
//     sum_i w_i * |T(p_i) - q_i|^2
//
// The checks below independently derive the optimum for:
//   * scans recorded in different length units (known uniform scale k),
//     with scaling allowed (exact recovery) and scale locked (rigid pose
//     whose residual is the analytic constrained minimum),
//   * zero-confidence outlier markers and integer-duplicated weights,
//   * mirrored marker layouts that no chirality-preserving rigid motion
//     can match exactly: the returned pose must be a legal proper
//     rotation and leave exactly the minimum achievable residual.
// ----------------------------------------------------------------------------

void
assertRegistrationNear (
    double       actual,
    double       expected,
    double       relTol,
    double       absTol,
    const char*  what)
{
    const double diff = std::abs (actual - expected);
    const double tol  = absTol + relTol * std::abs (expected);
    if (diff > tol)
    {
        std::cerr << "Procrustes registration mismatch: " << what
                  << "\n       actual   = " << actual
                  << "\n       expected = " << expected
                  << "\n       diff     = " << diff
                  << "\n       tol      = " << tol << std::endl;
        assert (diff <= tol);
    }
}

double
registrationMatrixMaxDiff (
    const IMATH_NAMESPACE::M44d& a,
    const IMATH_NAMESPACE::M44d& b)
{
    double maxDiff = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            maxDiff = std::max (maxDiff, std::abs (a[i][j] - b[i][j]));
    return maxDiff;
}

template <typename T>
IMATH_NAMESPACE::V3d
weightedMarkerCentroid (
    const std::vector<IMATH_NAMESPACE::Vec3<T>>& points,
    const std::vector<T>&                        weights)
{
    IMATH_NAMESPACE::V3d center (0.0, 0.0, 0.0);
    double               weightSum = 0.0;
    for (size_t i = 0; i < points.size (); ++i)
    {
        const double w = static_cast<double> (weights[i]);
        center += w * IMATH_NAMESPACE::V3d (points[i]);
        weightSum += w;
    }
    return center / weightSum;
}

// Length of one row of the upper-left 3x3 block; for a similarity transform
// every row has length |s|, the uniform scale.
inline double
registrationBlockScale (const IMATH_NAMESPACE::M44d& m)
{
    return std::sqrt (
        m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]);
}

// Verify that the upper-left block is s times a proper rotation:
// orthonormal rows and determinant +1 after normalization.  Returns s.
double
assertProperSimilarityPose (
    const IMATH_NAMESPACE::M44d& m,
    double                        orthoTol,
    const char*                   label)
{
    using namespace IMATH_NAMESPACE;

    const double s = registrationBlockScale (m);
    assert (s > 0.0);

    const double row1 = std::sqrt (
        m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]);
    const double row2 = std::sqrt (
        m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]);
    assertRegistrationNear (row1, s, orthoTol, orthoTol, label);
    assertRegistrationNear (row2, s, orthoTol, orthoTol, label);

    M33d r (
        m[0][0] / s, m[0][1] / s, m[0][2] / s,
        m[1][0] / s, m[1][1] / s, m[1][2] / s,
        m[2][0] / s, m[2][1] / s, m[2][2] / s);

    const M33d rrT = r * r.transposed ();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double expected = (i == j) ? 1.0 : 0.0;
            assertRegistrationNear (
                rrT[i][j], expected, orthoTol, orthoTol, label);
        }
    }

    // Chirality check: a reflection would give -1 and is not a legal
    // rigid/similarity pose for this registration.
    assertRegistrationNear (r.determinant (), 1.0, orthoTol, orthoTol, label);
    return s;
}

// The weighted centroids must correspond, which independently checks the
// translation against t = centerTo - centerFrom * Linear(T), including the
// scale/translation coupling.
template <typename T>
void
assertCentroidTranslation (
    const IMATH_NAMESPACE::M44d&                 m,
    const std::vector<IMATH_NAMESPACE::Vec3<T>>& from,
    const std::vector<IMATH_NAMESPACE::Vec3<T>>& to,
    const std::vector<T>&                         weights,
    double                                        tol,
    const char*                                   label)
{
    using namespace IMATH_NAMESPACE;

    const V3d fromCenter = weightedMarkerCentroid (from, weights);
    const V3d toCenter   = weightedMarkerCentroid (to, weights);

    const V3d mappedCenter = fromCenter * m;
    assertRegistrationNear (
        (mappedCenter - toCenter).length (), 0.0, 0.0, tol, label);

    const V3d fromCenterLinear (
        fromCenter.x * m[0][0] + fromCenter.y * m[1][0] +
            fromCenter.z * m[2][0],
        fromCenter.x * m[0][1] + fromCenter.y * m[1][1] +
            fromCenter.z * m[2][1],
        fromCenter.x * m[0][2] + fromCenter.y * m[1][2] +
            fromCenter.z * m[2][2]);
    const V3d expectedTranslation = toCenter - fromCenterLinear;

    assertRegistrationNear (
        m[3][0], expectedTranslation.x, 0.0, tol, label);
    assertRegistrationNear (
        m[3][1], expectedTranslation.y, 0.0, tol, label);
    assertRegistrationNear (
        m[3][2], expectedTranslation.z, 0.0, tol, label);
}

// Two scans using different length units.  Scan 1 is in inches, scan 2 in
// millimetres, so the true uniform scale is k = 25.4, followed by a known
// proper rotation and translation.  Confidences are uneven and one far
// marker has zero confidence.
template <typename T>
void
testDifferentLengthUnitsAndConfidences ()
{
    std::cout << "  different length units / confidence weights ("
              << (sizeof (T) == 8 ? "double" : "float") << ")... "
              << std::flush;

    using namespace IMATH_NAMESPACE;
    typedef Vec3<T> Vec;

    // Small integer coordinates are exactly representable in float as
    // well, keeping the analytic tolerances sharp in both precisions.
    std::vector<Vec> scanInches = {
        Vec (T (1), T (-1), T (0)),
        Vec (T (-2), T (1), T (1)),
        Vec (T (0), T (2), T (-1)),
        Vec (T (2), T (1), T (2)),
        Vec (T (-1), T (-2), T (2)),
        Vec (T (1), T (1), T (-2)),
        Vec (T (-1), T (3), T (1)),
        Vec (T (2), T (-2), T (-1)),
        Vec (T (100), T (-80), T (60)) // far outlier, confidence 0
    };
    std::vector<T> confidence = { T (3), T (1),   T (4),   T (2),  T (5),
                                  T (2), T (3),   T (4),   T (0) };
    const size_t   numMarkers = scanInches.size ();
    const size_t   trusted    = numMarkers - 1;

    const double k = 25.4; // inches -> millimetres

    M44d known;
    known.makeIdentity ();
    known.translate (V3d (4.2, -7.1, 13.5));
    known = known * Eulerd (0.37, -0.61, 0.83).toMatrix44 ();
    known.scale (V3d (k, k, k));

    std::vector<Vec> scanMillimetres (numMarkers);
    for (size_t i = 0; i < trusted; ++i)
        scanMillimetres[i] = Vec (V3d (scanInches[i]) * known);

    // The last marker has zero confidence and is a genuine outlier: its
    // reported second-scan position is badly wrong.
    scanMillimetres[trusted] = Vec (V3d (900.0, -700.0, 450.0));

    const V3d fromCenter =
        weightedMarkerCentroid (scanInches, confidence);

    // Weighted central second moment of the first scan:
    //     M2 = sum_i w_i |p_i - center_w|^2.
    double weightedM2 = 0.0;
    for (size_t i = 0; i < numMarkers; ++i)
    {
        const double w = static_cast<double> (confidence[i]);
        weightedM2 +=
            w * (V3d (scanInches[i]) - fromCenter).length2 ();
    }

    const double pointTol = sizeof (T) == 8 ? 1e-8 : 1e-4;
    const double analyticRelTol = sizeof (T) == 8 ? 1e-9 : 1e-4;
    const double matrixTol = 1e-10;

    // ---- Similarity registration: size and placement must be recovered.
    const M44d similarity = procrustesRotationAndTranslation (
        &scanInches[0],
        &scanMillimetres[0],
        &confidence[0],
        numMarkers,
        true);

    const double simScale = assertProperSimilarityPose (
        similarity, 1e-8, "unit similarity legality");
    assertRegistrationNear (
        simScale, k, 1e-9, 1e-7 * k, "recovered unit scale");

    for (size_t i = 0; i < trusted; ++i)
    {
        const V3d residual =
            V3d (scanInches[i]) * similarity - V3d (scanMillimetres[i]);
        assert (residual.length () < pointTol);
    }
    assertRegistrationNear (
        procrustesError (
            &scanInches[0],
            &scanMillimetres[0],
            &confidence[0],
            numMarkers,
            similarity),
        0.0,
        0.0,
        sizeof (T) == 8 ? 1e-14 : 1e-6,
        "similarity weighted residual");

    assertCentroidTranslation (
        similarity,
        scanInches,
        scanMillimetres,
        confidence,
        pointTol,
        "unit similarity translation");

    // ---- Rigid registration: scale is locked to 1.
    //
    // With q_i = k R p_i + t and the optimal proper R recovered exactly,
    // locking s = 1 leaves
    //     sum_i w_i |R(p_i-c) - k R(p_i-c)|^2
    //   = (k - 1)^2 * M2
    // as the global minimum residual.
    const M44d rigid = procrustesRotationAndTranslation (
        &scanInches[0],
        &scanMillimetres[0],
        &confidence[0],
        numMarkers,
        false);

    const double rigidScale = assertProperSimilarityPose (
        rigid, 1e-8, "unit rigid legality");
    assertRegistrationNear (rigidScale, 1.0, 1e-9, 1e-9, "locked scale");

    const double expectedRigidError = (k - 1.0) * (k - 1.0) * weightedM2;
    const double rigidError = procrustesError (
        &scanInches[0],
        &scanMillimetres[0],
        &confidence[0],
        numMarkers,
        rigid);
    assertRegistrationNear (
        rigidError,
        expectedRigidError,
        analyticRelTol,
        1e-8 * expectedRigidError,
        "rigid unit-mismatch minimum residual");

    assertCentroidTranslation (
        rigid,
        scanInches,
        scanMillimetres,
        confidence,
        pointTol,
        "unit rigid translation");

    // ---- Zero-confidence marker must be ignored everywhere.
    // Removing the zero-weight marker must give the identical pose: it
    // contributes to neither centroid, cross covariance, scale denominator
    // nor residual.
    const M44d similarityNoOutlier =
        procrustesRotationAndTranslation (
            &scanInches[0],
            &scanMillimetres[0],
            &confidence[0],
            trusted,
            true);
    const M44d rigidNoOutlier = procrustesRotationAndTranslation (
        &scanInches[0],
        &scanMillimetres[0],
        &confidence[0],
        trusted,
        false);

    assert (
        registrationMatrixMaxDiff (similarity, similarityNoOutlier) <
        matrixTol);
    assert (
        registrationMatrixMaxDiff (rigid, rigidNoOutlier) < matrixTol);

    // The far marker is a genuine outlier that biases a solve ignoring
    // confidence: its unweighted pose leaves more than twice the optimal
    // weighted rigid residual on the trusted markers (the locked-scale
    // term (k-1)^2 M2 is the common floor in that comparison).
    const M44d unweightedWithOutlier =
        procrustesRotationAndTranslation (
            &scanInches[0], &scanMillimetres[0], numMarkers, false);
    const double unweightedTrustedError = procrustesError (
        &scanInches[0],
        &scanMillimetres[0],
        &confidence[0],
        trusted,
        unweightedWithOutlier);
    assert (unweightedTrustedError > 2.0 * expectedRigidError);

    std::cout << "OK\n";
}

// Weights must act as per-marker replication counts: a marker with weight 4
// should produce the same pose as four identical copies of that marker.
template <typename T>
void
testConfidenceEquivalentToReplication ()
{
    std::cout << "  confidence-as-replication equivalence ("
              << (sizeof (T) == 8 ? "double" : "float") << ")... "
              << std::flush;

    using namespace IMATH_NAMESPACE;
    typedef Vec3<T> Vec;

    std::vector<Vec> from = {
        Vec (T (1), T (-1), T (0)),
        Vec (T (-2), T (1), T (1)),
        Vec (T (0), T (2), T (-1)),
        Vec (T (2), T (1), T (2))
    };

    M44d known;
    known.makeIdentity ();
    known.translate (V3d (-2.0, 5.0, 1.5));
    known = known * Eulerd (0.2, 0.9, -0.5).toMatrix44 ();
    known.scale (V3d (3.5, 3.5, 3.5));

    std::vector<Vec> to (from.size ());
    for (size_t i = 0; i < from.size (); ++i)
        to[i] = Vec (V3d (from[i]) * known);

    std::vector<T> weights = { T (1), T (1), T (1), T (4) };

    // Fourth marker replicated four times (so it appears four times total,
    // matching weight 4 together with the three unit-weight markers).
    std::vector<Vec> replicatedFrom = {
        from[0], from[1], from[2],
        from[3], from[3], from[3], from[3]
    };
    std::vector<Vec> replicatedTo = {
        to[0], to[1], to[2],
        to[3], to[3], to[3], to[3]
    };

    const double tol = sizeof (T) == 8 ? 1e-12 : 1e-6;

    const M44d weightedRigid = procrustesRotationAndTranslation (
        &from[0], &to[0], &weights[0], from.size (), false);
    const M44d replicatedRigid = procrustesRotationAndTranslation (
        &replicatedFrom[0],
        &replicatedTo[0],
        replicatedFrom.size (),
        false);
    assert (
        registrationMatrixMaxDiff (weightedRigid, replicatedRigid) < tol);

    const M44d weightedSimilarity = procrustesRotationAndTranslation (
        &from[0], &to[0], &weights[0], from.size (), true);
    const M44d replicatedSimilarity =
        procrustesRotationAndTranslation (
            &replicatedFrom[0],
            &replicatedTo[0],
            replicatedFrom.size (),
            true);
    assert (
        registrationMatrixMaxDiff (
            weightedSimilarity, replicatedSimilarity) < tol);

    std::cout << "OK\n";
}

// Mirrored marker distribution with distinct weighted variances along the
// three axes.  The target markers are an improper (chirality reversing)
// image of the source markers, so zero residual is only possible with an
// illegal reflection.  The proper solution must absorb the mismatch by
// reversing along the least-variance axis.
template <typename T>
void
testMirroredMarkersChiralityAndResidual ()
{
    std::cout << "  mirrored markers: chirality and constrained residual ("
              << (sizeof (T) == 8 ? "double" : "float") << ")... "
              << std::flush;

    using namespace IMATH_NAMESPACE;
    typedef Vec3<T> Vec;

    // Symmetric +/- pairs on the principal axes give an exactly diagonal
    // weighted covariance with clearly separated variances
    // (lambdaX=32, lambdaY=24, lambdaZ=4).  A common offset makes the
    // weighted centroid non-zero and moves it off the z=0 mirror plane.
    std::vector<Vec> centered = {
        Vec (T (4), T (0), T (0)),  Vec (T (-4), T (0), T (0)),
        Vec (T (0), T (2), T (0)),  Vec (T (0), T (-2), T (0)),
        Vec (T (0), T (0), T (1)),  Vec (T (0), T (0), T (-1))
    };
    std::vector<T> weights = { T (1), T (1), T (3), T (3), T (2), T (2) };
    const V3d      offset (2.0, -3.0, 1.0);

    std::vector<Vec> from (centered.size ());
    for (size_t i = 0; i < centered.size (); ++i)
        from[i] = Vec (V3d (centered[i]) + offset);

    const V3d fromCenter = weightedMarkerCentroid (from, weights);
    assertRegistrationNear (
        (fromCenter - offset).length (), 0.0, 0.0, 1e-12,
        "mirror weighted centroid");

    double lambda[3] = { 0.0, 0.0, 0.0 };
    double totalVariance = 0.0;
    for (size_t i = 0; i < from.size (); ++i)
    {
        const V3d d = V3d (from[i]) - fromCenter;
        const double w = static_cast<double> (weights[i]);
        lambda[0] += w * d.x * d.x;
        lambda[1] += w * d.y * d.y;
        lambda[2] += w * d.z * d.z;
        totalVariance += w * d.length2 ();
    }

    // The scenario only has a decidable flip direction if the three
    // variances are distinct and well separated.
    assert (lambda[0] > lambda[1] * 1.25);
    assert (lambda[1] > lambda[2] * 2.0);
    int minAxis = 0;
    for (int axis = 1; axis < 3; ++axis)
        if (lambda[axis] < lambda[minAxis]) minAxis = axis;
    const double lambdaMin = lambda[minAxis];

    // Improper transform: reflection of z, plus translation.  Mirror about
    // the global z=0 plane on purpose, so the weighted center (z = 1) does
    // not lie on the mirror plane.
    M44d reflection;
    reflection.makeIdentity ();
    reflection.scale (V3d (1.0, 1.0, -1.0));
    const V3d translation (5.0, -1.0, 4.0);

    std::vector<Vec> mirrored (from.size ());
    for (size_t i = 0; i < from.size (); ++i)
        mirrored[i] = Vec (V3d (from[i]) * reflection + translation);

    const double analyticRelTol = sizeof (T) == 8 ? 1e-9 : 1e-4;
    const double residualAbsTol = sizeof (T) == 8 ? 1e-10 : 1e-5;

    // ---- Scale locked: best legal rigid pose.
    //
    // Flipping the correction on axis j costs exactly 4 * lambda[j]; with
    // distinct variances the unique optimum is j = argmin lambda, so the
    // global minimum weighted residual is 4 * lambdaMin.
    const M44d rigid = procrustesRotationAndTranslation (
        &from[0], &mirrored[0], &weights[0], from.size (), false);

    const double rigidScale =
        assertProperSimilarityPose (rigid, 1e-8, "mirror rigid legality");
    assertRegistrationNear (rigidScale, 1.0, 1e-9, 1e-9, "mirror scale lock");

    assertRegistrationNear (
        procrustesError (
            &from[0], &mirrored[0], &weights[0], from.size (), rigid),
        4.0 * lambdaMin,
        analyticRelTol,
        residualAbsTol,
        "mirror rigid minimum residual");

    assertCentroidTranslation (
        rigid, from, mirrored, weights, 1e-8, "mirror rigid translation");

    // Decisive flip-direction check.  Probe the linear block on each
    // principal axis by transforming center +/- e_axis (translation
    // cancels).  The two high-variance axes must agree with the illegal
    // reflection; the least-variance axis must point the opposite way,
    // i.e. that is the singular direction the proper solve corrects.
    for (int axis = 0; axis < 3; ++axis)
    {
        V3d unit (0.0, 0.0, 0.0);
        unit[axis] = 1.0;
        const V3d mappedAxis =
            ((fromCenter + unit) * rigid - (fromCenter - unit) * rigid) *
            0.5;
        const V3d reflectedAxis = unit * reflection;
        const double agreement = mappedAxis.dot (reflectedAxis);
        if (axis == minAxis)
            assertRegistrationNear (
                agreement, -1.0, 1e-8, 1e-8, "corrected least-variance axis");
        else
            assertRegistrationNear (
                agreement, 1.0, 1e-8, 1e-8, "preserved high-variance axis");
    }

    // ---- Uniform scale allowed.
    //
    // With the same optimal proper rotation Q, optimizing s gives
    //     s* = 1 - 2 * lambdaMin / totalVariance
    // and
    //     E* = 4 * lambdaMin * (1 - lambdaMin / totalVariance).
    const M44d similarity = procrustesRotationAndTranslation (
        &from[0], &mirrored[0], &weights[0], from.size (), true);

    const double simScale = assertProperSimilarityPose (
        similarity, 1e-8, "mirror similarity legality");
    const double expectedScale =
        1.0 - 2.0 * lambdaMin / totalVariance;
    assertRegistrationNear (
        simScale, expectedScale, analyticRelTol, 1e-9,
        "mirror similarity optimum scale");

    assertRegistrationNear (
        similarity.determinant (),
        expectedScale * expectedScale * expectedScale,
        analyticRelTol,
        1e-10,
        "mirror similarity positive determinant");

    assertRegistrationNear (
        procrustesError (
            &from[0], &mirrored[0], &weights[0], from.size (), similarity),
        4.0 * lambdaMin * (1.0 - lambdaMin / totalVariance),
        analyticRelTol,
        residualAbsTol,
        "mirror similarity minimum residual");

    assertCentroidTranslation (
        similarity,
        from,
        mirrored,
        weights,
        1e-7,
        "mirror similarity translation");

    // ---- Same conclusion when the reflection is followed by a general
    // proper rotation: the chirality correction must still land on the
    // least-variance axis and leave the same residual.
    const M44d properRotation =
        Eulerd (0.5, 1.1, -0.4).toMatrix44 ();
    const M44d improperGeneral = reflection * properRotation;

    std::vector<Vec> rotatedMirrored (from.size ());
    for (size_t i = 0; i < from.size (); ++i)
        rotatedMirrored[i] =
            Vec (V3d (from[i]) * improperGeneral + V3d (-2, 7, 3));

    const M44d rotatedRigid = procrustesRotationAndTranslation (
        &from[0],
        &rotatedMirrored[0],
        &weights[0],
        from.size (),
        false);

    assertProperSimilarityPose (
        rotatedRigid, 1e-8, "rotated mirror legality");
    assertRegistrationNear (
        procrustesError (
            &from[0],
            &rotatedMirrored[0],
            &weights[0],
            from.size (),
            rotatedRigid),
        4.0 * lambdaMin,
        analyticRelTol,
        residualAbsTol,
        "rotated mirror minimum residual");

    for (int axis = 0; axis < 3; ++axis)
    {
        V3d unit (0.0, 0.0, 0.0);
        unit[axis] = 1.0;
        const V3d mappedAxis =
            ((fromCenter + unit) * rotatedRigid -
             (fromCenter - unit) * rotatedRigid) *
            0.5;
        const V3d improperAxis = unit * improperGeneral;
        const double agreement = mappedAxis.dot (improperAxis);
        if (axis == minAxis)
            assertRegistrationNear (
                agreement, -1.0, 1e-7, 1e-7,
                "rotated corrected least-variance axis");
        else
            assertRegistrationNear (
                agreement, 1.0, 1e-7, 1e-7,
                "rotated preserved variance axis");
    }

    std::cout << "OK\n";
}

template <typename T>
void
testProcrustesImp ()
{
    // Test the empty case:
    IMATH_NAMESPACE::M44d id = procrustesRotationAndTranslation (
        (IMATH_NAMESPACE::Vec3<T>*) 0,
        (IMATH_NAMESPACE::Vec3<T>*) 0,
        (T*) 0,
        0);
    assert (id == IMATH_NAMESPACE::M44d ());

    id = procrustesRotationAndTranslation (
        (IMATH_NAMESPACE::Vec3<T>*) 0,
        (IMATH_NAMESPACE::Vec3<T>*) 0,
        0);
    assert (id == IMATH_NAMESPACE::M44d ());

    // First we'll test with a bunch of known translation/rotation matrices
    // to make sure we get back exactly the same points:
    IMATH_NAMESPACE::M44d m;
    m.makeIdentity ();
    testTranslationRotationMatrix<T> (m);

    m.translate (IMATH_NAMESPACE::V3d (3.0, 5.0, -0.2));
    testTranslationRotationMatrix<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (M_PI, 0, 0));
    testTranslationRotationMatrix<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (0, M_PI / 4.0, 0));
    testTranslationRotationMatrix<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (0, 0, -3.0 / 4.0 * M_PI));
    testTranslationRotationMatrix<T> (m);

    m.makeIdentity ();
    testWithTranslateRotateAndScale<T> (m);

    m.translate (IMATH_NAMESPACE::V3d (0.4, 6.0, 10.0));
    testWithTranslateRotateAndScale<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (M_PI, 0, 0));
    testWithTranslateRotateAndScale<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (0, M_PI / 4.0, 0));
    testWithTranslateRotateAndScale<T> (m);

    m.rotate (IMATH_NAMESPACE::V3d (0, 0, -3.0 / 4.0 * M_PI));
    testWithTranslateRotateAndScale<T> (m);

    m.scale (IMATH_NAMESPACE::V3d (2.0, 2.0, 2.0));
    testWithTranslateRotateAndScale<T> (m);

    m.scale (IMATH_NAMESPACE::V3d (0.01, 0.01, 0.01));
    testWithTranslateRotateAndScale<T> (m);

    // Now we'll test with some random point sets and verify
    // the various Procrustes properties:
    std::vector<IMATH_NAMESPACE::Vec3<T>> fromPoints;
    std::vector<IMATH_NAMESPACE::Vec3<T>> toPoints;
    fromPoints.clear ();
    toPoints.clear ();

    for (size_t i = 0; i < 4; ++i)
    {
        const T theta = T (2 * i) / T (M_PI);
        const T offset = T(M_PI / 3.0);
        fromPoints.push_back (
            IMATH_NAMESPACE::Vec3<T> (cos (theta), sin (theta), 0));
        toPoints.push_back (IMATH_NAMESPACE::Vec3<T> (
            cos (theta + offset), sin (theta + offset), 0));
    }
    verifyProcrustes (fromPoints, toPoints);

    IMATH_NAMESPACE::Rand48 random (1209);
    for (size_t numPoints = 1; numPoints < 10; ++numPoints)
    {
        fromPoints.clear ();
        toPoints.clear ();
        for (size_t i = 0; i < numPoints; ++i)
        {
            fromPoints.push_back (IMATH_NAMESPACE::Vec3<T> (
                static_cast<T>(random.nextf ()),
                static_cast<T>(random.nextf ()),
                static_cast<T>(random.nextf ())));
            toPoints.push_back (IMATH_NAMESPACE::Vec3<T> (
                static_cast<T>(random.nextf ()),
                static_cast<T>(random.nextf ()),
                static_cast<T>(random.nextf ())));
        }
    }
    verifyProcrustes (fromPoints, toPoints);

    // Test with some known matrices of varying degrees of quality:
    testProcrustesWithMatrix<T> (m);

    m.translate (IMATH_NAMESPACE::Vec3<T> (3, 4, 1));
    testProcrustesWithMatrix<T> (m);

    m.translate (IMATH_NAMESPACE::Vec3<T> (-10, 2, 1));
    testProcrustesWithMatrix<T> (m);

    IMATH_NAMESPACE::Eulerd rot (M_PI / 3.0, 3.0 * M_PI / 4.0, 0);
    m = m * rot.toMatrix44 ();
    testProcrustesWithMatrix<T> (m);

    m.scale (IMATH_NAMESPACE::Vec3<T> (T(1.5), T(6.4), T(2.0)));
    testProcrustesWithMatrix<T> (m);

    IMATH_NAMESPACE::Eulerd rot2 (1.0, M_PI, M_PI / 3.0);
    m = m * rot.toMatrix44 ();

    m.scale (IMATH_NAMESPACE::Vec3<T> (-1, 1, 1));
    testProcrustesWithMatrix<T> (m);

    m.scale (IMATH_NAMESPACE::Vec3<T> (1, T(0.001), 1));
    testProcrustesWithMatrix<T> (m);

    m.scale (IMATH_NAMESPACE::Vec3<T> (1, 1, 0));
    testProcrustesWithMatrix<T> (m);

    // Scenario-based weighted registration / chirality tests:
    testDifferentLengthUnitsAndConfidences<T> ();
    testConfidenceEquivalentToReplication<T> ();
    testMirroredMarkersChiralityAndResidual<T> ();
}

void
testProcrustes ()
{
    std::cout << "Testing Procrustes algorithms in single precision..."
              << std::endl;
    testProcrustesImp<float> ();

    std::cout << "Testing Procrustes algorithms in double precision..."
              << std::endl;
    testProcrustesImp<double> ();
}
