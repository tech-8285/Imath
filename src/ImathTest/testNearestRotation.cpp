//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testNearestRotation.h"

#include <ImathEuler.h>
#include <ImathMatrix.h>
#include <ImathRandom.h>

#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace IMATH_NAMESPACE;

namespace
{

//
// Tolerances.  The matrix-level tolerances are deliberately tight: an
// operation that only rewrites the *representation* of a rotation must
// not move the camera by any measurable amount.
//

template <class T> T
matrixTolerance ();
template <> float
matrixTolerance<float> ()
{
    return 1e-5f;
}
template <> double
matrixTolerance<double> ()
{
    // Tight enough to catch a float round-trip inside Euler<double>.
    return 1e-9;
}

template <class T>
bool
matricesEqual (const Matrix33<T>& a, const Matrix33<T>& b, T tolerance)
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            if (std::fabs (a[r][c] - b[r][c]) > tolerance) return false;
    return true;
}

template <class T>
bool
vectorsEqual (const Vec3<T>& a, const Vec3<T>& b, T tolerance)
{
    return std::fabs (a.x - b.x) <= tolerance &&
           std::fabs (a.y - b.y) <= tolerance &&
           std::fabs (a.z - b.z) <= tolerance;
}

//
// All 24 legal rotation orders.
//
template <class T>
vector<typename Euler<T>::Order>
allOrders ()
{
    using E = Euler<T>;
    return {
        E::XYZ, E::XZY, E::YZX, E::YXZ, E::ZXY, E::ZYX,
        E::XZX, E::XYX, E::YXY, E::YZY, E::ZYZ, E::ZXZ,
        E::XYZr, E::XZYr, E::YZXr, E::YXZr, E::ZXYr, E::ZYXr,
        E::XZXr, E::XYXr, E::YXYr, E::YZYr, E::ZYZr, E::ZXZr
    };
}

//
// Return the alternate Euler triplet that encodes exactly the same
// orientation as `angles` but lives on the other side of the +/-PI
// branches.  Angles are in storage (ijk) order, matching the slot
// formulas derived from Euler<T>::toMatrix33():
//
//   non-repeating orders: (x - PI, PI - y, z - PI)
//   repeated-axis orders: (x - PI, -y,     z - PI)
//
// testBranchEquivalence() below verifies numerically that these triplets
// are truly orientation-equivalent before any other test relies on them.
//
template <class T>
Vec3<T>
equivalentBranchIJK (const Vec3<T>& angles, typename Euler<T>::Order order)
{
    const T  pi = T (M_PI);
    Euler<T> probe (0, 0, 0, order);
    Vec3<T>  other;

    if (probe.initialRepeated ())
    {
        other.x = angles.x - pi;
        other.y = -angles.y;
        other.z = angles.z - pi;
    }
    else
    {
        other.x = angles.x - pi;
        other.y = pi - angles.y;
        other.z = angles.z - pi;
    }
    return other;
}

//
// Same alternate branch, expressed in physical-axis (XYZ) space.  This is
// the layout nearestRotation() documents for its parameters.
//
template <class T>
Vec3<T>
equivalentBranchXYZ (const Vec3<T>& xyzAngles, typename Euler<T>::Order order)
{
    Euler<T> original (xyzAngles, order, Euler<T>::XYZLayout);
    Vec3<T>   branchIJK =
        equivalentBranchIJK<T> (Vec3<T> (original), order);
    Euler<T> alternate (branchIJK, order);
    return alternate.toXYZVector ();
}

//
// angleMod(): reduce an angle to the equivalent value in [-PI, PI].
//
template <class T>
void
testAngleMod ()
{
    using E = Euler<T>;

    const T pi = T (M_PI);

    assert (E::angleMod (T (0)) == T (0));
    assert (std::fabs (E::angleMod (T (1.5)) - T (1.5)) < T (1e-6));
    assert (std::fabs (E::angleMod (T (-1.5)) - T (-1.5)) < T (1e-6));

    // Crossing the +PI branch: 3/2 PI wraps to -1/2 PI.
    assert (std::fabs (E::angleMod (T (1.5) * pi) + T (0.5) * pi) < T (1e-5));
    // Crossing the -PI branch: -3/2 PI wraps to +1/2 PI.
    assert (std::fabs (E::angleMod (T (-1.5) * pi) - T (0.5) * pi) <
            T (1e-5));

    // Full turns must disappear.  An angle sitting exactly on the branch
    // boundary may legitimately reduce to either +PI or -PI: both are the
    // same point on the unit circle, so accept whichever boundary is hit.
    auto boundaryDistance = [pi] (T reduced) {
        return std::min (std::fabs (reduced - pi),
                         std::fabs (reduced + pi));
    };
    assert (boundaryDistance (E::angleMod (T (7) * pi)) < T (1e-5));
    assert (boundaryDistance (E::angleMod (T (-7) * pi)) < T (1e-5));

    // The result always lies on [-PI, PI].
    Rand48 rand (156);
    for (int i = 0; i < 5000; ++i)
    {
        T a       = T (rand.nextf (-1000.0, 1000.0));
        T reduced = E::angleMod (a);
        assert (reduced >= -pi - T (1e-6));
        assert (reduced <= pi + T (1e-6));
    }

    // Over a moderate range (where trig argument reduction is exact),
    // the reduced angle names the same point on the unit circle.
    for (int i = 0; i < 5000; ++i)
    {
        T a       = T (rand.nextf (-4.0 * M_PI, 4.0 * M_PI));
        T reduced = E::angleMod (a);
        assert (std::fabs (std::cos (reduced) - std::cos (a)) < T (1e-5));
        assert (std::fabs (std::sin (reduced) - std::sin (a)) < T (1e-5));
    }

    //
    // Precision contract: for Euler<double> the reduced value must keep
    // double precision.  angleMod() is currently declared to return
    // float, which truncates this result; the assertion pins down the
    // required behavior.
    //
    if (std::is_same<T, double>::value)
    {
        const double x = 123456.7890123456789;
        double expected = std::fmod (x, 2.0 * M_PI);
        if (expected < -M_PI) expected += 2.0 * M_PI;
        if (expected > M_PI) expected -= 2.0 * M_PI;
        const double got = E::angleMod (T (x));
        assert (std::fabs (got - expected) < 1e-12);
    }

    cout << "angleMod" << (std::is_same<T, double>::value ? " (double)"
                                                           : " (float)")
         << ": ok" << endl;
}

//
// simpleXYZRotation(): every component is pulled to within +/-PI of the
// corresponding target component, without changing any orientation when
// used with an XYZ Euler.
//
template <class T>
void
testSimpleXYZRotation ()
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    const T pi = T (M_PI);

    V3 target (T (0.2), T (-0.3), T (0.4));
    V3 rot (target.x + T (4) * pi, target.y - T (2) * pi,
            target.z + T (6) * pi);

    const Matrix33<T> before = E (rot, E::XYZ).toMatrix33 ();
    E::simpleXYZRotation (rot, target);
    const Matrix33<T> after = E (rot, E::XYZ).toMatrix33 ();

    assert (matricesEqual (before, after, matrixTolerance<T> ()));

    const V3 d = rot - target;
    assert (std::fabs (d.x) <= pi + T (1e-6));
    assert (std::fabs (d.y) <= pi + T (1e-6));
    assert (std::fabs (d.z) <= pi + T (1e-6));

    cout << "simpleXYZRotation"
         << (std::is_same<T, double>::value ? " (double)" : " (float)")
         << ": ok" << endl;
}

//
// Guard for the test helper itself: the alternate branch triplet must be
// orientation-equivalent to the original for every one of the 24 orders.
//
template <class T>
void
testBranchEquivalence ()
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    Rand48 rand (31337);

    for (typename E::Order order : allOrders<T> ())
    {
        for (int i = 0; i < 500; ++i)
        {
            // Keep the middle angle away from the gimbal-lock poles.
            V3 angles (T (rand.nextf (-3.0, 3.0)),
                       T (rand.nextf (-1.4, 1.4)),
                       T (rand.nextf (-3.0, 3.0)));

            // Helper in storage (ijk) space.
            E original (angles, order);
            E alternate (
                equivalentBranchIJK<T> (angles, order), order);

            if (!matricesEqual (
                    original.toMatrix33 (),
                    alternate.toMatrix33 (),
                    matrixTolerance<T> ()))
            {
                cout << "branch helper is not orientation-equivalent for "
                        "order 0x"
                     << hex << int (order) << dec << endl;
                assert (false);
            }

            // Same property through the physical-axis (XYZ) helper.
            V3 xyz = original.toXYZVector ();
            E  alternateXyz (
                equivalentBranchXYZ<T> (xyz, order), order, E::XYZLayout);

            if (!matricesEqual (
                    original.toMatrix33 (),
                    alternateXyz.toMatrix33 (),
                    matrixTolerance<T> ()))
            {
                cout << "XYZ branch helper is not orientation-equivalent "
                        "for order 0x"
                     << hex << int (order) << dec << endl;
                assert (false);
            }
        }
    }

    cout << "equivalent-branch helper (24 orders)"
         << (std::is_same<T, double>::value ? " (double)" : " (float)")
         << ": ok" << endl;
}

//
// Property 1: unwrapping may only change the representation, never the
// spatial orientation.  A frame stored on the far equivalent branch (and
// offset by extra full turns) must be brought next to the target while
// its rotation matrix stays byte-for-byte equivalent, for every rotation
// order, repeated-axis orders included.
//
template <class T>
void
testNearestRotationPreservesOrientation ()
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    const T pi = T (M_PI);
    Rand48 rand (20240101);

    for (typename E::Order order : allOrders<T> ())
    {
        for (int i = 0; i < 250; ++i)
        {
            // Angles are authored in physical-axis (XYZ) layout.
            V3 targetXyz (T (rand.nextf (-2.5, 2.5)),
                          T (rand.nextf (-1.3, 1.3)),
                          T (rand.nextf (-2.5, 2.5)));
            E target (targetXyz, order, E::XYZLayout);

            // Same camera orientation, recorded on the other branch and
            // pushed several full turns away from the target.
            V3 currentXyz = equivalentBranchXYZ<T> (targetXyz, order);
            currentXyz.x += T (2) * pi;
            currentXyz.z -= T (2) * pi;

            const Matrix33<T> orientationBefore =
                E (currentXyz, order, E::XYZLayout).toMatrix33 ();

            V3 adjustedXyz = currentXyz;
            E::nearestRotation (adjustedXyz, targetXyz, order);

            const Matrix33<T> orientationAfter =
                E (adjustedXyz, order, E::XYZLayout).toMatrix33 ();

            if (!matricesEqual (
                    orientationBefore,
                    orientationAfter,
                    matrixTolerance<T> ()))
            {
                cout << "nearestRotation changed the camera orientation "
                        "for order 0x"
                     << hex << int (order) << dec << endl;
                assert (false);
            }

            // The adjusted curve must actually be continuous relative to
            // the target: no component may still sit a branch away.
            const V3 gap = adjustedXyz - targetXyz;
            if (std::fabs (gap.x) > pi + T (1e-5) ||
                std::fabs (gap.y) > pi + T (1e-5) ||
                std::fabs (gap.z) > pi + T (1e-5))
            {
                cout << "nearestRotation left a branch discontinuity for "
                        "order 0x"
                     << hex << int (order) << dec << " gap=" << gap << endl;
                assert (false);
            }
        }
    }

    cout << "nearestRotation orientation + continuity (24 orders)"
         << (std::is_same<T, double>::value ? " (double)" : " (float)")
         << ": ok" << endl;
}

//
// Property 1, animated-curve form: a camera sweeping continuously
// through the +PI branch must not develop a jump in the angle curve, and
// every adjusted keyframe must keep pointing along the animated path.
// Exercises both a non-repeating order and a repeated-axis order.
//
template <class T>
void
testBranchCrossingAlongMotion (typename Euler<T>::Order order, const char* label)
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    const T pi = T (M_PI);

    // First physical-axis angle starts just below PI and sweeps past it;
    // the other two angles drift gently, as they would on a real move.
    vector<V3> keyframes;
    for (int frame = 0; frame <= 16; ++frame)
    {
        T s = T (frame);
        keyframes.push_back (
            V3 (T (3.05) + T (0.03) * s,
                T (0.35) + T (0.01) * s,
                T (0.2) - T (0.02) * s));
    }

    E previous (keyframes[0], order, E::XYZLayout);

    for (size_t frame = 1; frame < keyframes.size (); ++frame)
    {
        E current (keyframes[frame], order, E::XYZLayout);
        const Matrix33<T> animatedOrientation = current.toMatrix33 ();

        V3 angles         = current.toXYZVector ();
        V3 previousAngles = previous.toXYZVector ();
        E::nearestRotation (angles, previousAngles, order);
        E adjusted (angles, order, E::XYZLayout);

        // The camera must still face the orientation the animation gave
        // this frame, regardless of where the angles were unwrapped to.
        if (!matricesEqual (
                adjusted.toMatrix33 (),
                animatedOrientation,
                matrixTolerance<T> ()))
        {
            cout << label << ": orientation lost at frame " << frame << endl;
            assert (false);
        }

        // No frame-to-frame branch jump.
        const V3 step = adjusted.toXYZVector () - previous.toXYZVector ();
        if (std::fabs (step.x) >= pi || std::fabs (step.y) >= pi ||
            std::fabs (step.z) >= pi)
        {
            cout << label << ": angle curve jumps a branch at frame "
                 << frame << " step=" << step << endl;
            assert (false);
        }

        previous = adjusted;
    }

    cout << label << ": ok" << endl;
}

//
// Property 2: neighboring keyframes may use different rotation orders to
// record the same camera placement.  makeNear() must remove the angle
// difference that is purely a recording artifact while:
//
//   * preserving the spatial orientation of the adjusted frame,
//   * preserving that frame's original rotation order,
//   * leaving the target frame untouched.
//
template <class T>
void
testMakeNearMixedOrders ()
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    Rand48 rand (0xC0FFEE);
    const auto orders = allOrders<T> ();

    for (int sample = 0; sample < 3000; ++sample)
    {
        typename E::Order targetOrder =
            orders[size_t (rand.nextf (0, float (orders.size ()) - 0.001f))];
        typename E::Order frameOrder =
            orders[size_t (rand.nextf (0, float (orders.size ()) - 0.001f))];

        V3 targetAngles (T (rand.nextf (-2.0, 2.0)),
                         T (rand.nextf (-1.2, 1.2)),
                         T (rand.nextf (-2.0, 2.0)));

        E target (targetAngles, targetOrder);

        // The neighboring frame records the SAME camera orientation in a
        // possibly different rotation order.
        E frame (target.toMatrix33 (), frameOrder);

        const typename E::Order keptOrder = frame.order ();
        const Matrix33<T>    keptOrientation = frame.toMatrix33 ();
        const E              targetSnapshot   = target;

        frame.makeNear (target);

        // Original rotation order is retained.
        if (frame.order () != keptOrder)
        {
            cout << "makeNear rewrote the frame's rotation order" << endl;
            assert (false);
        }

        // The camera orientation is retained.
        if (!matricesEqual (
                frame.toMatrix33 (),
                keptOrientation,
                matrixTolerance<T> ()))
        {
            cout << "makeNear changed the orientation of a frame "
                    "(orders 0x"
                 << hex << int (targetOrder) << " -> 0x" << int (frameOrder)
                 << dec << ")" << endl;
            assert (false);
        }

        // The target keyframe is an input and must not be modified.
        assert (target.order () == targetSnapshot.order ());
        assert (vectorsEqual (
            V3 (target), V3 (targetSnapshot), T (1e-12)));
    }

    cout << "makeNear mixed-order preservation (24x24 order pairs)"
         << (std::is_same<T, double>::value ? " (double)" : " (float)")
         << ": ok" << endl;
}

//
// Property 2, discrimination check:
//
//   * Two frames describing the same orientation in different orders
//     carry only a representational difference; once both are expressed
//     in a common order after makeNear(), their angles must agree.
//
//   * Two frames describing genuinely different orientations carry a
//     real camera rotation; makeNear() must not erase it.
//
template <class T>
void
testRepresentationDifferenceVsRealRotation ()
{
    using E  = Euler<T>;
    using V3 = Vec3<T>;

    const T pi = T (M_PI);

    //
    // Same camera, all 24 recording orders.
    //
    const V3 referenceAngles (T (0.42), T (-0.55), T (0.71));
    const E  reference (referenceAngles, E::ZYX);

    for (typename E::Order order : allOrders<T> ())
    {
        E target (reference.toMatrix33 (), E::XYZ);
        E frame (reference.toMatrix33 (), order);

        frame.makeNear (target);

        // Re-express both frames in the same order: the purely
        // representational angle difference must have vanished.
        const E frameInCommon (frame.toMatrix33 (), E::XYZ);
        const V3 gap = V3 (frameInCommon) - V3 (target);

        if (!vectorsEqual (gap, V3 (0), T (1e-4)))
        {
            cout << "representational angle difference survived makeNear "
                    "for order 0x"
                 << hex << int (order) << dec << " gap=" << gap << endl;
            assert (false);
        }
    }

    //
    // A genuine camera rotation must survive makeNear intact.
    //
    E target (V3 (T (0.2), T (0.1), T (-0.3)), E::ZYX);
    E rotated (V3 (T (0.9), T (-0.7), T (0.4)), E::YXZ);

    const Matrix33<T> rotatedOrientation = rotated.toMatrix33 ();
    rotated.makeNear (target);

    assert (matricesEqual (
        rotated.toMatrix33 (),
        rotatedOrientation,
        matrixTolerance<T> ()));

    // The relative rotation between the two frames is unchanged and is
    // plainly not the identity.
    const Matrix33<T> relative =
        rotated.toMatrix33 () * target.toMatrix33 ().transposed ();
    const T trace =
        relative[0][0] + relative[1][1] + relative[2][2];
    assert (std::fabs (trace - T (3)) > T (0.5));

    // The angle gap to the target is nevertheless unwrapped so the
    // curve itself stays continuous.
    const E rotatedInTargetOrder (rotated.toMatrix33 (), E::ZYX);
    const V3 gap = V3 (rotatedInTargetOrder) - V3 (target);
    assert (std::fabs (gap.x) <= pi + T (1e-5));
    assert (std::fabs (gap.y) <= pi + T (1e-5));
    assert (std::fabs (gap.z) <= pi + T (1e-5));

    cout << "representation difference vs. real camera rotation"
         << (std::is_same<T, double>::value ? " (double)" : " (float)")
         << ": ok" << endl;
}

template <class T>
void
runForType (const char* typeName)
{
    cout << "== " << typeName << " ==" << endl;

    testAngleMod<T> ();
    testSimpleXYZRotation<T> ();
    testBranchEquivalence<T> ();
    testNearestRotationPreservesOrientation<T> ();

    testBranchCrossingAlongMotion<T> (Euler<T>::XYZ, "branch crossing, XYZ");
    testBranchCrossingAlongMotion<T> (Euler<T>::ZYZ, "branch crossing, ZYZ");

    testMakeNearMixedOrders<T> ();
    testRepresentationDifferenceVsRealRotation<T> ();
}

} // namespace

void
testNearestRotation ()
{
    cout << "Testing nearest Euler rotation (animation-curve continuity)"
         << endl;

    runForType<float> ("Eulerf");
    runForType<double> ("Eulerd");

    cout << "ok\n" << endl;
}
