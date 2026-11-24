//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A spotlight is mounted on a rotating parent node.  Animation keyframes
// record the world-space direction the lamp points at, while at playback
// the lamp head rotates in the parent node's local space.
//
// These tests exercise the two conventions that make such a rig behave:
//
//   * shortest-arc rotation -- slerpShortestArc() and Quat::setRotation()
//     always take the smaller of the two arcs between two orientations or
//     two directions, and a quaternion and its negation describe the same
//     orientation (the double cover of SO(3));
//
//   * rigid-body hierarchy transforms -- the head's world orientation is
//     the parent's world orientation composed with the head's local
//     orientation (q_world = q_local * q_parent), so changing the scene's
//     reference frame re-expresses, but does not change, every local pose.
//
// The lamp's rest pose carries an installation twist: a side marker on the
// beam cone is rotated about the optical axis.  The tests check that the
// beam and that side marker stay rigidly attached as the rig moves.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testSpotLightRig.h"

#include <ImathFun.h>
#include <ImathMatrix.h>
#include <ImathQuat.h>
#include <ImathRandom.h>
#include <ImathVecAlgo.h>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;
using namespace IMATH_NAMESPACE;

namespace
{

const float EPS      = 1.0e-5f;
const float LOOSE_EPS = 1.0e-4f;

//
// The lamp head, modeled in the rotating parent's local space.
//
// beamAxis is the rest direction of the light cone (local +z); sideAxis is
// a marker on the rim of the spot, rotated by the installation twist around
// the optical axis, and therefore perpendicular to the beam.
//
struct LampRig
{
    V3f beamAxis;
    V3f sideAxis;
};

//
// Compare two directions.
//
void
checkVec (const V3f& a, const V3f& b, float e = EPS)
{
    assert (a.equalWithAbsError (b, e));
}

//
// Compare two orientations, accounting for the double cover: q and -q are
// the same rotation.
//
void
checkQuat (const Quatf& a, const Quatf& b, float e = EPS)
{
    float sign = (a ^ b) < 0.0f ? -1.0f : 1.0f;
    assert (equalWithAbsError (a.r, sign * b.r, e));
    assert (a.v.equalWithAbsError (sign * b.v, e));
}

//
// Unsigned geodesic angle between two orientations, in [0, pi].
//
float
orientationAngle (const Quatf& a, const Quatf& b)
{
    float d = min (1.0f, fabs (a ^ b));
    return 2.0f * acos (d);
}

//
// Bake a world-space aim direction, recorded at a keyframe, into the head's
// local orientation for a given parent pose.
//
// setRotation() builds the shortest-arc rotation that carries the rest beam
// onto the requested direction.  Because the target direction alone carries
// no twist, the installation twist travels with the head rigidly: this is
// the unique shortest rotation, and it leaves the side marker perpendicular
// to the new beam.
//
Quatf
bakeLocalKey (
    const LampRig& rig,
    const Quatf&   parentWorld,
    const V3f&     worldAim)
{
    V3f localAim =
        worldAim * parentWorld.inverse ().toMatrix44 ();

    Quatf local;
    local.setRotation (rig.beamAxis, localAim);
    return local;
}

//
// World-space beam and side-marker directions for a local head pose.
//
V3f
worldBeam (const LampRig& rig, const Quatf& parentWorld, const Quatf& local)
{
    return (rig.beamAxis * local) * parentWorld;
}

V3f
worldSide (const LampRig& rig, const Quatf& parentWorld, const Quatf& local)
{
    return (rig.sideAxis * local) * parentWorld;
}

//
// Build the lamp described in the file header: rest beam along +z, with a
// non-trivial installation twist about the optical axis.
//
LampRig
makeRig ()
{
    const float twist = 35.0f * float(M_PI) / 180.0f;

    LampRig rig;
    rig.beamAxis = V3f (0, 0, 1);

    Quatf twistRot;
    twistRot.setAxisAngle (rig.beamAxis, twist);
    rig.sideAxis = V3f (1, 0, 0) * twistRot;

    // The rest frame is an orthonormal pair.
    assert (fabs (rig.sideAxis ^ rig.beamAxis) < EPS);
    assert (equalWithAbsError (rig.sideAxis.length (), 1.0f, EPS));

    return rig;
}

//
// A parent orientation and two distinct world-space aim directions, shared
// by the first three test groups.
//
void
makeScene (
    Quatf& parent0,
    Quatf& parent1,
    V3f&  aim0,
    V3f&  aim1)
{
    parent0.setAxisAngle (V3f (0, 1, 0), 0.30f);
    parent1.setAxisAngle (V3f (0, 1, 0), 1.10f);

    aim0 = V3f (1.0f, 0.2f, 0.5f).normalized ();
    aim1 = V3f (0.4f, -1.0f, 0.7f).normalized ();
}

//
// (1) Changing the scene's reference frame rotates the beam and the side
//     marker with the scene, while leaving every baked local pose and the
//     local playback path untouched.
//
void
testReferenceFrameChange (const LampRig& rig)
{
    cout << "  reference-frame change rotates beam and marker, keeps local "
            "poses"
         << endl;

    Quatf parent0, parent1;
    V3f   aim0, aim1;
    makeScene (parent0, parent1, aim0, aim1);

    Quatf local0 = bakeLocalKey (rig, parent0, aim0);
    Quatf local1 = bakeLocalKey (rig, parent1, aim1);

    // An arbitrary rotation of the whole scene's reference coordinates.
    Quatf frame;
    frame.setAxisAngle (V3f (0.3f, 0.8f, 0.5f).normalized (), 0.9f);

    // Re-express the parent and the recorded aims in the new frame.
    Quatf parent0r = frame * parent0;
    Quatf parent1r = frame * parent1;
    V3f  aim0r    = aim0 * frame;
    V3f  aim1r    = aim1 * frame;

    Quatf local0r = bakeLocalKey (rig, parent0r, aim0r);
    Quatf local1r = bakeLocalKey (rig, parent1r, aim1r);

    // Local keyframes are independent of the scene frame chosen.
    checkQuat (local0, local0r);
    checkQuat (local1, local1r);

    // At each keyframe the world beam aims at the recorded target.
    checkVec (worldBeam (rig, parent0, local0), aim0);
    checkVec (worldBeam (rig, parent1, local1), aim1);
    checkVec (worldBeam (rig, parent0r, local0r), aim0r);

    // The beam and the side marker simply rotate with the reference frame.
    checkVec (worldBeam (rig, parent0r, local0r),
              worldBeam (rig, parent0, local0) * frame);
    checkVec (worldSide (rig, parent0r, local0r),
              worldSide (rig, parent0, local0) * frame);
    checkVec (worldBeam (rig, parent1r, local1r),
              worldBeam (rig, parent1, local1) * frame);
    checkVec (worldSide (rig, parent1r, local1r),
              worldSide (rig, parent1, local1) * frame);

    // The marker stays rigidly perpendicular to the beam in every frame.
    assert (fabs (worldSide (rig, parent0r, local0r) ^
                  worldBeam (rig, parent0r, local0r)) < EPS);

    //
    // Rigid-body hierarchy through matrices, including a mount offset.
    // Both setTranslation() and setAxisAngle() reset the matrix, so the
    // translation and rotation are composed by multiplication.
    //
    V3f mountOffset (0.5f, -0.25f, 0.8f);
    M44f tMat, rMat;
    tMat.setTranslation (mountOffset);
    rMat.setAxisAngle (V3f (0, 1, 0), 0.30f);
    M44f parentMatrix = rMat * tMat;

    M44f localMatrix = local0.toMatrix44 ();
    M44f worldMatrix = localMatrix * parentMatrix;

    // Directions ignore translation and agree with the quaternion chain.
    V3f matrixBeam;
    worldMatrix.multDirMatrix (rig.beamAxis, matrixBeam);
    checkVec (matrixBeam, worldBeam (rig, parent0, local0));

    // The head's origin is carried to the parent's mount point.
    V3f headOrigin;
    worldMatrix.multVecMatrix (V3f (0), headOrigin);
    checkVec (headOrigin, mountOffset);

    //
    // The inverse hierarchy brings world directions back down the chain.
    // Rotation-only matrices are used so translation does not enter a
    // direction transform.
    //
    M44f parentRot = parent0.toMatrix44 ();
    M44f localRot  = local0.toMatrix44 ();
    M44f rotationWorld = localRot * parentRot;

    // Removing the full world transform returns the rest-frame beam.
    V3f recoveredRest;
    rotationWorld.inverse ().multDirMatrix (aim0, recoveredRest);
    checkVec (recoveredRest, rig.beamAxis);

    // Removing only the parent transform returns the head-local aim used
    // when the keyframe was baked.
    V3f recoveredAim;
    parentRot.inverse ().multDirMatrix (aim0, recoveredAim);
    checkVec (recoveredAim, rig.beamAxis * local0);

    //
    // Randomized re-framing: the local pose and the marker/beam
    // relationship must hold for any scene rotation.
    //
    Rand48 rand (5);
    for (int i = 0; i < 200; ++i)
    {
        V3f axis = hollowSphereRand<V3f> (rand);
        Quatf anyFrame;
        anyFrame.setAxisAngle (axis, rand.nextf (-float(M_PI), float(M_PI)));

        Quatf pr = anyFrame * parent0;
        V3f  ar = aim0 * anyFrame;
        Quatf lr = bakeLocalKey (rig, pr, ar);

        checkQuat (lr, local0, LOOSE_EPS);
        checkVec (worldBeam (rig, pr, lr), ar, LOOSE_EPS);
        checkVec (worldSide (rig, pr, lr),
                  worldSide (rig, parent0, local0) * anyFrame,
                  LOOSE_EPS);
    }
}

//
// (2) Adjacent keyframes are played back along the shortest arc between
//     the local poses.  Equivalent keyframe representations must trace the
//     same spot path.
//
void
testEquivalentKeyframesAndShortestArc (const LampRig& rig)
{
    cout << "  shortest-arc playback; equivalent keyframes trace one path"
         << endl;

    Quatf parent0, parent1;
    V3f   aim0, aim1;
    makeScene (parent0, parent1, aim0, aim1);

    Quatf q0 = bakeLocalKey (rig, parent0, aim0);
    Quatf q1 = bakeLocalKey (rig, parent1, aim1);

    // The negated pair is the identical set of orientations.
    Quatf n0 = -q0;
    Quatf n1 = -q1;
    checkQuat (q0, n0);
    checkQuat (q1, n1);

    const int steps = 32;

    //
    // The two representations produce the same orientation and the same
    // moving spot (beam and side marker) at every playback sample.
    //
    for (int i = 0; i <= steps; ++i)
    {
        float t = float(i) / steps;

        Quatf a = slerpShortestArc (q0, q1, t);
        Quatf b = slerpShortestArc (n0, n1, t);
        checkQuat (a, b);

        V3f spotA = rig.sideAxis * a;
        V3f spotB = rig.sideAxis * b;
        checkVec (spotA, spotB);

        V3f beamA = rig.beamAxis * a;
        V3f beamB = rig.beamAxis * b;
        checkVec (beamA, beamB);
    }

    //
    // slerpShortestArc() is an explicit end-point flip followed by the
    // plain slerp(); the two formulations agree sample for sample.
    //
    for (int i = 0; i <= steps; ++i)
    {
        float t = float(i) / steps;
        Quatf shortest = slerpShortestArc (q0, q1, t);
        Quatf manual =
            (q0 ^ q1) >= 0.0f ? slerp (q0, q1, t) : slerp (q0, -q1, t);
        checkQuat (shortest, manual);
    }

    //
    // Shortest-arc motion is a constant-axis great circle: the geodesic
    // angle grows in equal increments that never exceed pi.
    //
    float total = orientationAngle (q0, q1);
    assert (total <= float(M_PI) + EPS);

    float maxStep = 0.0f;
    Quatf previous = q0;
    for (int i = 1; i <= 8; ++i)
    {
        Quatf q = slerpShortestArc (q0, q1, float(i) / 8.0f);
        float step = orientationAngle (previous, q);
        maxStep   = max (maxStep, step);
        assert (equalWithAbsError (step, total / 8.0f, LOOSE_EPS));
        previous = q;
    }
    assert (maxStep < float(M_PI));

    //
    // The midpoint orientation applied to the rest beam equals the beam
    // produced by rotating the endpoint beam halfway along its arc.  The
    // midpoint quaternion is the normalized sum of the endpoint
    // quaternions (after the shortest-arc sign choice), which is exactly
    // what slerpShortestArc returns at t = 1/2.
    //
    Quatf midQuat = slerpShortestArc (q0, q1, 0.5f);
    V3f  midBeam  = rig.beamAxis * midQuat;

    Quatf chordQuat = (q0 + ((q0 ^ q1) >= 0.0f ? q1 : -q1)).normalized ();
    V3f  chordBeam  = rig.beamAxis * chordQuat;
    checkVec (midBeam, chordBeam, LOOSE_EPS);

    // The midpoint beam lies on the short great-circle arc between the
    // endpoint beams and is equidistant from both.
    V3f beamStart = rig.beamAxis * q0;
    V3f beamEnd   = rig.beamAxis * q1;
    assert (equalWithAbsError (
        midBeam ^ beamStart, midBeam ^ beamEnd, LOOSE_EPS));

    //
    // When the 4D dot of the endpoints is negative, plain slerp() takes
    // the long way around while slerpShortestArc() flips the second key.
    // Construct such a pair directly (the second rotation passes pi, so
    // the quaternion half-angles land in opposite hemispheres).
    //
    Quatf A;
    A.setAxisAngle (V3f (0, 0, 1), 0.3f);
    Quatf B;
    B.setAxisAngle (V3f (0, 0, 1), 3.6f);
    assert ((A ^ B) < 0.0f);

    auto pathLength = [&] (bool shortest) {
        float length = 0.0f;
        Quatf prev   = A;
        for (int i = 1; i <= 200; ++i)
        {
            Quatf q = shortest ? slerpShortestArc (A, B, float(i) / 200.0f)
                               : slerp (A, B, float(i) / 200.0f);
            length += angle4D (prev, q);
            prev    = q;
        }
        return length;
    };

    float shortLength = pathLength (true);
    float longLength  = pathLength (false);
    assert (shortLength < longLength);
    assert (shortLength < float(M_PI));
}

//
// (3) When the target passes behind the lamp, two consecutive keyframes
//     hold opposite directions.  A pair of opposite directions alone cannot
//     decide the head's full orientation: any axis perpendicular to the
//     beam flips one direction into the other by a pi rotation.
//
//     Neighboring keyframes remove that freedom: the motion axis is fixed by
//     the adjacent frames, and that axis determines which of the many
//     pi-rotations the head performs.
//
void
testAntipodalDirection ()
{
    cout << "  opposite directions: underdetermined alone, resolved by "
            "neighbors"
         << endl;

    //
    // (3a) Opposite directions alone: a whole circle of flip axes, each a
    //      valid head orientation that sends d to -d but turns the side
    //      marker to a different place.
    //
    V3f d = V3f (1, 2, 3).normalized ();

    // Two orthonormal vectors spanning the plane perpendicular to d.
    V3f e = V3f (0.4f, -1.0f, 0.2f);
    e      = orthogonal (d, e).normalized ();
    V3f f  = (d % e).normalized ();
    assert (fabs (e ^ d) < EPS);
    assert (fabs (f ^ d) < EPS);

    Quatf flipAboutE;
    flipAboutE.setAxisAngle (e, float(M_PI));
    Quatf flipAboutF;
    flipAboutF.setAxisAngle (f, float(M_PI));

    // Both satisfy the beam constraint d -> -d ...
    checkVec (d * flipAboutE, -d);
    checkVec (d * flipAboutF, -d);

    // ... yet they are different orientations, and each leaves its own flip
    // axis fixed while moving the side marker differently.
    assert (flipAboutE != flipAboutF);
    checkVec (e * flipAboutE, e);
    checkVec (f * flipAboutF, f);
    assert (!(e * flipAboutE).equalWithAbsError (e * flipAboutF, EPS));

    // Sweep the full circle of flip axes perpendicular to d: every axis is a
    // valid solution, and the rim marker lands in distinct positions, which
    // is exactly the ambiguity that two opposite directions cannot resolve.
    int distinctMarkers = 0;
    V3f previousMarker (9, 9, 9);
    for (int k = 0; k < 8; ++k)
    {
        float angle = k * float(M_PI) / 8.0f;
        V3f axis =
            (e * cos (angle) + f * sin (angle)).normalized ();

        Quatf flip;
        flip.setAxisAngle (axis, float(M_PI));
        checkVec (d * flip, -d);

        V3f marker = e * flip;
        if ((marker - previousMarker).length () > 0.5f) ++distinctMarkers;
        previousMarker = marker;
    }
    assert (distinctMarkers > 1);

    //
    // (3b) Neighbor keyframes determine the rotation axis.  The head turns
    //      about an axis perpendicular to the rest beam, so keys separated
    //      by pi in angle aim at exactly opposite directions.
    //
    V3f restBeam (0, 0, 1);
    V3f axis = V3f (0.3f, -0.7f, 0.0f).normalized ();
    assert (fabs (axis ^ restBeam) < EPS);

    Quatf k0;
    k0.setAxisAngle (axis, -float(M_PI_2));
    Quatf k1; // identity: the target is directly ahead
    Quatf k2;
    k2.setAxisAngle (axis, float(M_PI_2));
    Quatf k3;
    k3.setAxisAngle (axis, float(M_PI));

    V3f d0 = restBeam * k0;
    V3f d1 = restBeam * k1;
    V3f d2 = restBeam * k2;
    V3f d3 = restBeam * k3;

    // The two outer keys aim in opposite directions.
    checkVec (d0, -d2);
    checkVec (d1, -d3);

    // The motion axis is recoverable from either neighboring direction pair.
    V3f axisFromNeighbors = (d0 % d1).normalized ();
    checkVec (axisFromNeighbors, axis);

    // An axis different from the neighbor-derived one still flips d0 to d2,
    // demonstrating that the antipodal pair on its own admits other heads.
    V3f other = axis + 0.3f * d1;
    other     = orthogonal (d0, other).normalized ();
    assert (!other.equalWithAbsError (axis, EPS));
    Quatf otherFlip;
    otherFlip.setAxisAngle (other, float(M_PI));
    checkVec (d0 * otherFlip, d2);

    //
    // Playback across the antipodal key.  A single segment from k0 to k2
    // interpolates between antipodal quaternions, whose midpoint is not
    // uniquely defined -- that is the same ambiguity shown in (3a).  The
    // head is instead driven through the intermediate neighbor key k1, and
    // the two well-defined segments join on one consistent orientation.
    //
    checkQuat (slerpShortestArc (k0, k1, 1.0f),
               slerpShortestArc (k1, k2, 0.0f));
    checkQuat (slerpShortestArc (k1, k2, 1.0f),
               slerpShortestArc (k2, k3, 0.0f));

    // The neighbor-resolved route reaches k1 exactly, the orientation that
    // aims the beam straight ahead between the two opposite outer keys.
    checkQuat (slerpShortestArc (k0, k1, 1.0f), k1);
    checkVec (restBeam * slerpShortestArc (k0, k1, 1.0f), d1);

    //
    // A target that passes behind the lamp without landing on an exact
    // antipode stays well defined throughout: take aims just short of the
    // antipode and confirm the beam sweeps the short arc, staying a unit
    // direction and never jumping to the opposite side of the sphere.
    //
    const float nearAngle = float(M_PI_2) - 0.05f;
    Quatf near0, near2;
    near0.setAxisAngle (axis, -nearAngle);
    near2.setAxisAngle (axis, nearAngle);
    V3f nearStart = restBeam * near0;
    V3f nearEnd   = restBeam * near2;
    for (int i = 0; i <= 20; ++i)
    {
        Quatf q = slerpShortestArc (near0, near2, float(i) / 20.0f);
        V3f  beam = restBeam * q;

        assert (equalWithAbsError (beam.length (), 1.0f, EPS));

        // Every intermediate beam stays on the same hemisphere as the
        // midpoint, i.e. it never takes the long way around.
        V3f chordMid = (nearStart + nearEnd).normalized ();
        assert ((beam ^ chordMid) >= -LOOSE_EPS);
    }

    // The whole four-key path advances in uniform shortest-arc increments.
    Quatf keys[4] = {k0, k1, k2, k3};
    for (int segment = 0; segment < 3; ++segment)
    {
        float worst = 0.0f;
        for (int i = 1; i <= 10; ++i)
        {
            Quatf q = slerpShortestArc (
                keys[segment], keys[segment + 1], float(i) / 10.0f);
            Quatf p = slerpShortestArc (
                keys[segment], keys[segment + 1], float(i - 1) / 10.0f);

            worst = max (
                worst,
                fabs (orientationAngle (p, q) - float(M_PI_2) / 10.0f));

            // The beam stays a unit direction rigidly carried by the head.
            V3f beam = restBeam * q;
            assert (equalWithAbsError (beam.length (), 1.0f, EPS));
        }
        assert (worst < LOOSE_EPS);
    }
}

} // namespace

void
testSpotLightRig ()
{
    cout << "Testing spotlight rig: shortest-arc rotation and rigid-body "
            "hierarchy"
         << endl;

    LampRig rig = makeRig ();

    testReferenceFrameChange (rig);
    testEquivalentKeyframesAndShortestArc (rig);
    testAntipodalDirection ();

    cout << "ok\n" << endl;
}
