//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Tests for the "nearest rotation" utilities used when an animation curve
// stores successive camera orientations as Euler angles:
//
//    Euler<T>::angleMod()
//    Euler<T>::simpleXYZRotation()
//    Euler<T>::nearestRotation()
//    Euler<T>::makeNear()
//
// The contract under test is the one an animation editor relies on:
//
//  1. Continuity of the angle curve is only allowed to be judged while the
//     spatial orientation (rotation matrix / quaternion) is preserved.
//     Unwrapping an angle across a +/-PI branch may only pick another
//     Euler triplet that describes the very same orientation.  This holds
//     for every rotation order, including the repeated-axis orders
//     (XYX, XZX, YXY, YZY, ZXZ, ZYZ and their "relative" counterparts).
//
//  2. Two adjacent keyframes may record the same camera orientation using
//     different rotation orders.  The angle differences caused purely by
//     the choice of representation must be distinguishable from a real
//     rotation of the camera.  makeNear() must leave each keyframe's
//     original rotation order untouched.
//
// Note: this file encodes the required behavior.  On the unpatched
// Imath 3.2 code it exposes two defects, which the failing assertions
// identify precisely:
//
//    a) nearestRotation() builds its alternate Euler branch with the
//       XYZ/ZYX slot formula regardless of the requested order.  The
//       formula is only orientation-preserving for XYZ, ZYX, XYZr and
//       ZYXr; for the other twenty orders (all repeated-axis orders
//       included) it rotates the camera instead of merely unwrapping
//       the angles.
//    b) Euler<T>::angleMod() is declared to return float, so an
//       Euler<double> curve loses roughly 1e-7 radians of precision on
//       every unwrap.
//

#ifndef INCLUDED_TESTNEARESTROTATION_H
#define INCLUDED_TESTNEARESTROTATION_H

void testNearestRotation ();

#endif // INCLUDED_TESTNEARESTROTATION_H
