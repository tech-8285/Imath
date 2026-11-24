//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Tests for local framing of a perspective camera through
// Frustum<T>::window():
//
// A tiled renderer enlarges one screen-space tile of an off-axis
// perspective camera into its own image.  A pick ray cast through the
// enlarged image must still pass through the scene location that the
// original image pointed at.
//
// The conventions under test are those of the camera's near clipping
// plane window and the perspective projection:
//   * the eye is at the origin looking down -Z;
//   * screen space is NDC, -1 <= x,y <= 1;
//   * screenToLocal() maps an NDC point onto the near plane window
//     (left/right/bottom/top in camera-local coordinates);
//   * a perspective pick ray runs from the eye through that local
//     near-plane point (Frustum::projectScreenToRay());
//   * window(l,r,t,b) returns a frustum whose near-plane window is the
//     given screen rectangle in the parent frustum's local space.
//
// Everything is exercised from the public Frustum API only.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testFrustumWindow.h"

#include <ImathFrustum.h>
#include <ImathFun.h>
#include <ImathLine.h>
#include <ImathVec.h>

#include <assert.h>
#include <iostream>
#include <vector>

namespace
{

using IMATH_NAMESPACE::Frustum;
using IMATH_NAMESPACE::Line3;
using IMATH_NAMESPACE::Vec2;
using IMATH_NAMESPACE::Vec3;

//
// Tolerances, specialized per scalar type.
//

template <class T> T
pointAbsEps ();
template <>
inline float
pointAbsEps<float> ()
{
    return 1.0e-5f;
}
template <>
inline double
pointAbsEps<double> ()
{
    return 1.0e-12;
}

template <class T> T
rayDirEps ();
template <>
inline float
rayDirEps<float> ()
{
    return 1.0e-5f;
}
template <>
inline double
rayDirEps<double> ()
{
    return 1.0e-12;
}

//
// Component-wise comparison of two scene points that lie at the given
// camera-space depth.  Two pick rays sharing the eye can only diverge by
// their angular error, so the positional tolerance is an absolute floor
// plus rayDirEps scaled by the depth.  This bounds every component by the
// same geometric quantity regardless of its own magnitude (components
// near zero cannot be judged relative to themselves).
//

template <class T> bool
pointsAtDepthClose (const Vec3<T>& a, const Vec3<T>& b, T depth)
{
    const T tol =
        pointAbsEps<T> () + rayDirEps<T> () * std::max (depth, T (1));
    for (int i = 0; i < 3; ++i)
        if (IMATH_NAMESPACE::abs (a[i] - b[i]) > tol) return false;
    return true;
}

template <class T> bool
screenPointsClose (const Vec2<T>& a, const Vec2<T>& b)
{
    return a.equalWithAbsError (b, pointAbsEps<T> ());
}

//
// A screen rectangle in NDC: left, right, bottom, top with
// -1 <= l < r <= 1 and -1 <= b < t <= 1.
//

struct ScreenRect
{
    double l, r, b, t;
};

//
// Map an NDC point (u, v) of a sub-frustum back into the NDC of the
// parent frustum whose screen-space window is `rect`.  This is the
// inverse of Frustum::window(): both frusta share the same near-plane
// parameterization, so the mapping is the affine NDC remap.
//

template <class T> Vec2<T>
subNdcToParentNdc (const Vec2<T>& uv, const ScreenRect& rect)
{
    return Vec2<T> (
        T (rect.l) + (T (rect.r) - T (rect.l)) * (uv.x + T (1)) / T (2),
        T (rect.b) + (T (rect.t) - T (rect.b)) * (uv.y + T (1)) / T (2));
}

//
// Point on a perspective pick ray at the given camera-space depth
// (distance along -Z).  The eye is at the origin, so the ray through
// near-plane point p hits camera z = -depth at eye + dir * t with
// t = depth / (-dir.z).
//

template <class T> Vec3<T>
pointAtDepth (const Line3<T>& ray, T depth)
{
    const T t = depth / -ray.dir.z;
    return ray (t);
}

//
// Two perspective pick rays coincide within the visible depth range:
// they share the eye, their (normalized) directions agree, and points
// sampled on them between the near and far planes agree.
//

template <class T> void
assertRaysCoincide (
    const Line3<T>& ray0,
    const Line3<T>& ray1,
    T               nearPlane,
    T               farPlane)
{
    // Both perspective rays originate at the eye (the origin).
    assert (ray0.pos.equalWithAbsError (Vec3<T> (0, 0, 0), pointAbsEps<T> ()));
    assert (ray1.pos.equalWithAbsError (Vec3<T> (0, 0, 0), pointAbsEps<T> ()));

    // The directions are unit length; one minus their dot product is
    // the squared sine of the angle between them.
    const T dot   = ray0.dir ^ ray1.dir;
    const T scale = std::max (T (1), IMATH_NAMESPACE::abs (dot));
    assert ((T (1) - dot) <= rayDirEps<T> () * scale);

    // Coincidence must hold throughout the visible depth interval,
    // not only at the near plane.
    const T depths[] = {
        nearPlane,
        T (2) * nearPlane,
        T (10) * nearPlane,
        (nearPlane + farPlane) / T (2),
        farPlane - T (1.0e-3) * (farPlane - nearPlane)};

    for (T depth : depths)
    {
        if (depth <= nearPlane || depth >= farPlane) continue;

        const Vec3<T> p0 = pointAtDepth (ray0, depth);
        const Vec3<T> p1 = pointAtDepth (ray1, depth);

        assert (pointsAtDepthClose (p0, p1, depth));
    }
}

//
// Sample an NDC grid covering the whole [-1,1]^2 screen of a frustum.
//

std::vector<Vec2<double>>
ndcGrid ()
{
    const double                us[] = {-1.0, -0.63, -0.21, 0.18, 0.57, 1.0};
    const double                vs[] = {-1.0, -0.74, -0.33, 0.12, 0.61, 1.0};
    std::vector<Vec2<double>> points;
    for (double v : vs)
        for (double u : us)
            points.emplace_back (u, v);
    return points;
}

//
// Near-plane window convention: the sub-frustum's left/right/bottom/top
// must be exactly the parent's near-plane window coordinates of the
// rectangle's NDC corners.
//

template <class T> void
assertNearPlaneWindowConvention (
    const Frustum<T>& base, const Frustum<T>& sub, const ScreenRect& rect)
{
    const T lx =
        base.left () + (base.right () - base.left ()) * (T (rect.l) + 1) / 2;
    const T rx =
        base.left () + (base.right () - base.left ()) * (T (rect.r) + 1) / 2;
    const T by =
        base.bottom () + (base.top () - base.bottom ()) * (T (rect.b) + 1) / 2;
    const T ty =
        base.bottom () + (base.top () - base.bottom ()) * (T (rect.t) + 1) / 2;

    assert (IMATH_NAMESPACE::abs (sub.left () - lx) <= pointAbsEps<T> ());
    assert (IMATH_NAMESPACE::abs (sub.right () - rx) <= pointAbsEps<T> ());
    assert (IMATH_NAMESPACE::abs (sub.bottom () - by) <= pointAbsEps<T> ());
    assert (IMATH_NAMESPACE::abs (sub.top () - ty) <= pointAbsEps<T> ());

    // Local framing never changes the depth range or the projection type.
    assert (sub.nearPlane () == base.nearPlane ());
    assert (sub.farPlane () == base.farPlane ());
    assert (sub.orthographic () == base.orthographic ());
    assert (!sub.orthographic ());
}

//
// Round-trip picking:
//
// For every pick position on the original screen and every sub-window,
//   1. the scene point picked on the original image projects back onto
//      the original image at the same NDC position;
//   2. the pick ray built at the corresponding position in the enlarged
//      image coincides with the original pick ray over the whole visible
//      depth range, hence passes through the same scene point;
//   3. the scene point projects into the enlarged image at the position
//      the user picked, and mapping that position back through the local
//      window reproduces the original pick position.
//

template <class T> void
testRoundTripPick (const Frustum<T>& base, const std::vector<ScreenRect>& rects)
{
    for (const ScreenRect& rect : rects)
    {
        const Frustum<T> sub =
            base.window (T (rect.l), T (rect.r), T (rect.t), T (rect.b));

        assertNearPlaneWindowConvention (base, sub, rect);

        for (const Vec2<double>& uvD : ndcGrid ())
        {
            const Vec2<T> uv (uvD.x, uvD.y);

            // Position of this magnified-image pick in the original image.
            const Vec2<T> sBase = subNdcToParentNdc (uv, rect);

            const Line3<T> rayBase = base.projectScreenToRay (sBase);
            const Line3<T> raySub  = sub.projectScreenToRay (uv);

            assertRaysCoincide (rayBase, raySub, base.nearPlane (),
                                base.farPlane ());

            // A scene point on the picked ray, at an arbitrary visible depth.
            const T        depth = T (0.37) * (base.farPlane () -
                                                base.nearPlane ()) +
                            base.nearPlane ();
            const Vec3<T> scenePoint = pointAtDepth (rayBase, depth);

            // Projected back through each frustum, it lands on the pick.
            const Vec2<T> backBase = base.projectPointToScreen (scenePoint);
            const Vec2<T> backSub  = sub.projectPointToScreen (scenePoint);

            assert (screenPointsClose (backBase, sBase));
            assert (screenPointsClose (backSub, uv));

            // Projecting the magnified-image pick back through the local
            // window gives the original-image pick position.
            const Vec2<T> remapped = subNdcToParentNdc (backSub, rect);
            assert (screenPointsClose (remapped, sBase));
            assert (screenPointsClose (remapped, backBase));

            // The ray from the enlarged image literally contains the point.
            assert (pointsAtDepthClose (raySub.closestPointTo (scenePoint),
                                        scenePoint, depth));
        }
    }
}

//
// Two ways of framing the same region must give the same frustum:
//
//   (a) frame window W1 on the original image, then keep framing and
//       pick a second window W2 inside the enlarged image;
//   (b) pick the same region directly on the original image by
//       composing W2 into W1's screen rectangle.
//
// The resulting frusta are identical and their pick rays coincide at
// every screen position and every visible depth.
//

template <class T> void
testNestedAndDirectFraming (const Frustum<T>& base)
{
    const struct
    {
        ScreenRect w1;
        ScreenRect w2;
    } cases[] = {
        // Off-center parent window, arbitrary interior child window.
        {{-0.70, 0.40, -0.50, 0.80}, {-0.30, 0.60, -0.80, 0.25}},
        // Right-half parent, interior child with a different aspect.
        {{0.00, 1.00, -1.00, 1.00}, {-0.90, 0.90, -0.40, 0.70}},
        // Re-framing the whole enlarged image reproduces the parent.
        {{-0.55, 0.85, -0.65, 0.35}, {-1.00, 1.00, -1.00, 1.00}},
        // Nested windows with different position and aspect at every step.
        {{-1.00, -0.10, 0.10, 1.00}, {0.20, 0.95, -0.95, -0.10}}};

    for (const auto& c : cases)
    {
        const ScreenRect& w1 = c.w1;
        const ScreenRect& w2 = c.w2;

        // (a) Frame on the enlarged image.
        const Frustum<T> first =
            base.window (T (w1.l), T (w1.r), T (w1.t), T (w1.b));
        const Frustum<T> nested =
            first.window (T (w2.l), T (w2.r), T (w2.t), T (w2.b));

        // (b) Compose the two NDC rectangles and frame directly.
        const ScreenRect direct{
            subNdcToParentNdc (Vec2<T> (w2.l, w2.b), w1).x,
            subNdcToParentNdc (Vec2<T> (w2.r, w2.t), w1).x,
            subNdcToParentNdc (Vec2<T> (w2.l, w2.b), w1).y,
            subNdcToParentNdc (Vec2<T> (w2.r, w2.t), w1).y};

        const Frustum<T> directFrustum = base.window (
            T (direct.l), T (direct.r), T (direct.t), T (direct.b));

        // Same near-plane window, depth range and projection type.
        assert (nested.nearPlane () == directFrustum.nearPlane ());
        assert (nested.farPlane () == directFrustum.farPlane ());
        assert (nested.orthographic () == directFrustum.orthographic ());
        assert (IMATH_NAMESPACE::abs (nested.left () -
                                      directFrustum.left ()) <=
                pointAbsEps<T> ());
        assert (IMATH_NAMESPACE::abs (nested.right () -
                                      directFrustum.right ()) <=
                pointAbsEps<T> ());
        assert (IMATH_NAMESPACE::abs (nested.bottom () -
                                      directFrustum.bottom ()) <=
                pointAbsEps<T> ());
        assert (IMATH_NAMESPACE::abs (nested.top () - directFrustum.top ()) <=
                pointAbsEps<T> ());

        // The two framing routes describe the same window.  Exact
        // equality is not asserted: Frustum::screenToLocal() evaluates
        // its NDC remap with single-precision 1.f/2.f literals even for
        // double frusta, so the nested and direct computation chains
        // differ by last-place rounding (~1e-7 float, ~1e-16 double).
        assert (pointsAtDepthClose (
            Vec3<T> (nested.left (), nested.bottom (), nested.top ()),
            Vec3<T> (directFrustum.left (), directFrustum.bottom (),
                     directFrustum.top ()),
            T (1)));
        assert (IMATH_NAMESPACE::abs (nested.right () -
                                      directFrustum.right ()) <=
                pointAbsEps<T> ());

        // Every pick on the twice-enlarged image matches a pick made on
        // the directly-framed image, at every visible depth.
        for (const Vec2<double>& uvD : ndcGrid ())
        {
            const Vec2<T> uv (uvD.x, uvD.y);
            assertRaysCoincide (nested.projectScreenToRay (uv),
                                directFrustum.projectScreenToRay (uv),
                                base.nearPlane (), base.farPlane ());

            // And both still point at the scene location reached through
            // the original image.
            const Vec2<T>  baseNdc = subNdcToParentNdc (
                subNdcToParentNdc (uv, w2), w1);
            const Line3<T> baseRay = base.projectScreenToRay (baseNdc);
            assertRaysCoincide (baseRay, nested.projectScreenToRay (uv),
                                base.nearPlane (), base.farPlane ());
        }
    }
}

//
// Tiled rendering: the screen is partitioned into adjacent tiles of
// arbitrary position and aspect ratio, each enlarged into its own image.
// Rays from neighboring tiles at their shared edge must coincide, and
// scene positions across the seam must vary continuously.
//

template <class T> void
testTileSeams (const Frustum<T>& base)
{
    // Non-uniform, non-square partition of the full NDC screen.
    const T xBounds[] = {T (-1), T (-0.63), T (-0.07), T (0.41), T (1)};
    const T yBounds[] = {T (-1), T (-0.28), T (0.55), T (1)};

    const int nx = 4;
    const int ny = 3;

    Frustum<T> tiles[4][3];
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            tiles[i][j] = base.window (
                xBounds[i], xBounds[i + 1], yBounds[j + 1], yBounds[j]);

    const T seamVs[] = {T (-1), T (-0.53), T (0.04), T (0.62), T (1)};
    const T seamUs[] = {T (-1), T (-0.41), T (0.17), T (0.88), T (1)};

    // Vertical seams: the right edge (u = +1) of the left tile and the
    // left edge (u = -1) of the right tile are the same screen line.
    for (int i = 0; i < nx - 1; ++i)
    {
        for (T v : seamVs)
        {
            // The same tile-local v addresses the same screen height in
            // every tile of the row, since all tiles span the row's full
            // y interval.
            for (int j = 0; j < ny; ++j)
            {
                assertRaysCoincide (
                    tiles[i][j].projectScreenToRay (Vec2<T> (1, v)),
                    tiles[i + 1][j].projectScreenToRay (Vec2<T> (-1, v)),
                    base.nearPlane (), base.farPlane ());
            }
        }
    }

    // Horizontal seams: the top edge (v = +1) of the lower tile and the
    // bottom edge (v = -1) of the upper tile are the same screen line.
    for (int j = 0; j < ny - 1; ++j)
    {
        for (T u : seamUs)
        {
            for (int i = 0; i < nx; ++i)
            {
                assertRaysCoincide (
                    tiles[i][j].projectScreenToRay (Vec2<T> (u, 1)),
                    tiles[i][j + 1].projectScreenToRay (Vec2<T> (u, -1)),
                    base.nearPlane (), base.farPlane ());
            }
        }
    }

    // Continuity across a seam: stepping a small amount to either side
    // of the shared screen line gives two distinct, correctly ordered
    // scene points, and the gap matches what the parent frustum predicts.
    const T delta = T (0.02);
    for (int i = 0; i < nx - 1; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            const T v = T (0.31);

            const Vec2<T> leftNdc (T (1) - delta, v);
            const Vec2<T> rightNdc (T (-1) + delta, v);

            // Where those tile-local picks land in the parent image.
            const ScreenRect leftRect{
                xBounds[i], xBounds[i + 1], yBounds[j], yBounds[j + 1]};
            const ScreenRect rightRect{
                xBounds[i + 1], xBounds[i + 2], yBounds[j], yBounds[j + 1]};

            const Vec2<T> parentLeft =
                subNdcToParentNdc (leftNdc, leftRect);
            const Vec2<T> parentRight =
                subNdcToParentNdc (rightNdc, rightRect);

            // Both hug the shared boundary from their own side.
            assert (parentLeft.x < xBounds[i + 1]);
            assert (parentRight.x > xBounds[i + 1]);
            assert (IMATH_NAMESPACE::abs (parentLeft.x - xBounds[i + 1]) <
                    delta + pointAbsEps<T> ());
            assert (IMATH_NAMESPACE::abs (parentRight.x - xBounds[i + 1]) <
                    delta + pointAbsEps<T> ());

            const T depth = T (0.5) * (base.nearPlane () + base.farPlane ());
            const Vec3<T> leftPoint = pointAtDepth (
                tiles[i][j].projectScreenToRay (leftNdc), depth);
            const Vec3<T> rightPoint = pointAtDepth (
                tiles[i + 1][j].projectScreenToRay (rightNdc), depth);

            // Ordering on screen is preserved in the scene ...
            assert (leftPoint.x < rightPoint.x);

            // ... and each tile ray coincides with the parent-frustom ray
            // for the remapped NDC position, so the scene positions join
            // continuously with the un-tiled image.
            assert (pointsAtDepthClose (
                leftPoint,
                pointAtDepth (base.projectScreenToRay (parentLeft), depth),
                depth));
            assert (pointsAtDepthClose (
                rightPoint,
                pointAtDepth (base.projectScreenToRay (parentRight), depth),
                depth));
        }
    }

    // The four tiles meeting at the interior corner
    // (xBounds[2], yBounds[2]) all point at the same scene location
    // through that shared corner, addressed by each tile's own corner
    // NDC: lower-left tile takes its upper-right corner (+1,+1), etc.
    const Line3<T> cornerRays[] = {
        tiles[1][1].projectScreenToRay (Vec2<T> (1, 1)),
        tiles[2][1].projectScreenToRay (Vec2<T> (-1, 1)),
        tiles[1][2].projectScreenToRay (Vec2<T> (1, -1)),
        tiles[2][2].projectScreenToRay (Vec2<T> (-1, -1))};
    for (const Line3<T>& ray : cornerRays)
        assertRaysCoincide (cornerRays[0], ray, base.nearPlane (),
                            base.farPlane ());
}

//
// Pick positions on the enlarged image that correspond to the original
// image corners and center, including the degenerate full-screen window,
// for several off-axis frusta.
//

template <class T> void
testCornerAndCenterPicks (const Frustum<T>& base,
                          const std::vector<ScreenRect>& rects)
{
    const Vec2<T> anchors[] = {
        Vec2<T> (-1, -1), Vec2<T> (1, -1), Vec2<T> (-1, 1),
        Vec2<T> (1, 1),   Vec2<T> (0, 0),  Vec2<T> (0.37, -0.58)};

    for (const ScreenRect& rect : rects)
    {
        const Frustum<T> sub =
            base.window (T (rect.l), T (rect.r), T (rect.t), T (rect.b));

        for (const Vec2<T>& uv : anchors)
        {
            const Vec2<T>  sBase = subNdcToParentNdc (uv, rect);
            const Line3<T> a     = sub.projectScreenToRay (uv);
            const Line3<T> b     = base.projectScreenToRay (sBase);
            assertRaysCoincide (a, b, base.nearPlane (), base.farPlane ());
        }
    }
}

template <class T> void
runAll (const Frustum<T>& base, const std::vector<ScreenRect>& rects)
{
    testRoundTripPick (base, rects);
    testNestedAndDirectFraming (base);
    testTileSeams (base);
    testCornerAndCenterPicks (base, rects);
}

} // namespace

void
testFrustumWindow ()
{
    std::cout << "Testing local framing through Frustum::window()";

    // Screen rectangles exercising every position and aspect ratio:
    // off-center, wide-short, tall-thin, corner-touching, tiny, a full
    // screen tile, and a half screen.
    const std::vector<ScreenRect> rects = {
        {-0.85, 0.55, -0.70, 0.90},
        {-0.20, 0.80, -0.95, -0.15},
        {-0.95, -0.35, 0.05, 0.90},
        {-1.00, 0.00, -1.00, 0.00},
        {0.25, 0.75, 0.25, 0.75},
        {-1.00, 1.00, -1.00, 1.00},
        {-0.03, 0.02, -0.04, 0.01},
        {0.00, 1.00, -1.00, 1.00}};

    // Off-axis perspective frustum (asymmetric near-plane window).
    std::cout << "\nfloat off-axis ";
    {
        Frustum<float> base (
            1.7f, 567.0f, -3.5f, 2.0f, 0.9f, -1.3f, false);
        runAll (base, rects);
    }
    std::cout << "1";

    // Strongly off-axis: the whole near-plane window lies to one side.
    std::cout << "\nfloat strongly off-axis ";
    {
        Frustum<float> base (
            0.5f, 100.0f, 0.2f, 4.0f, -0.1f, -3.0f, false);
        runAll (base, rects);
    }
    std::cout << "2";

    // Symmetric frustum as a sanity reference.
    std::cout << "\nfloat symmetric ";
    {
        Frustum<float> base (1.0f, 1000.0f, -1.0f, 1.0f, 1.0f, -1.0f, false);
        runAll (base, rects);
    }
    std::cout << "3";

    // Double precision with the same off-axis geometry.
    std::cout << "\ndouble off-axis ";
    {
        Frustum<double> base (
            1.7, 567.0, -3.5, 2.0, 0.9, -1.3, false);
        runAll (base, rects);
    }
    std::cout << "4";

    std::cout << "\nok\n\n";
}
