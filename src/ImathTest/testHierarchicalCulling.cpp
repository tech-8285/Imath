//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testHierarchicalCulling.h"

#include <ImathBox.h>
#include <ImathFrustum.h>
#include <ImathFrustumTest.h>
#include <ImathMatrix.h>
#include <ImathSphere.h>
#include <ImathVec.h>

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace IMATH_NAMESPACE;

namespace
{

typedef Vec3<float>      V3;
typedef Box<Vec3<float>> B3;
typedef Sphere3<float>   S3;

enum NodeId
{
    P_BOX_PARENT = 0,
    P_BOX_INSIDE,
    P_BOX_EDGE_VISIBLE,
    P_BOX_OUTSIDE,
    P_BOX_CONSERVATIVE,

    P_SPHERE_PARENT,
    P_SPHERE_INSIDE,
    P_SPHERE_EDGE_VISIBLE,
    P_SPHERE_OUTSIDE,
    P_SPHERE_CONSERVATIVE,

    P_WHOLE_ROOT,
    P_WHOLE_BRANCH,
    P_WHOLE_NESTED_LEAF,
    P_WHOLE_DIRECT_LEAF,

    O_BOX_PARENT,
    O_BOX_INSIDE,
    O_BOX_EDGE_VISIBLE,
    O_BOX_OUTSIDE,
    O_BOX_CONSERVATIVE,

    O_SPHERE_PARENT,
    O_SPHERE_INSIDE,
    O_SPHERE_EDGE_VISIBLE,
    O_SPHERE_OUTSIDE,
    O_SPHERE_CONSERVATIVE,

    O_WHOLE_ROOT,
    O_WHOLE_BRANCH,
    O_WHOLE_NESTED_LEAF,
    O_WHOLE_DIRECT_LEAF,

    NODE_COUNT
};

enum CameraKind
{
    PERSPECTIVE_FRAME,
    MOVED_ORTHOGRAPHIC_FRAME
};

enum Decision
{
    UNSEEN,
    CULLED_BY_PARENT_BOX,
    PARTIAL_EXPAND_CHILDREN,
    DRAW_VISIBLE_LEAF,
    ACCEPT_WHOLE_SUBTREE
};

struct Node
{
    std::string      name;
    B3               box;
    S3               sphere;
    bool             usesSphere   = false;
    bool             conservative = false;
    std::vector<int> children;
    std::vector<V3>  samples;
};

struct Scene
{
    std::vector<Node> nodes;
    std::vector<int>  roots;

    Scene () : nodes (NODE_COUNT) {}

    Node&       node (int id) { return nodes[id]; }
    const Node& node (int id) const { return nodes[id]; }

    std::string name (int id) const
    {
        if (id >= 0 && id < NODE_COUNT && !nodes[id].name.empty ())
            return nodes[id].name;
        std::ostringstream out;
        out << "node-" << id;
        return out.str ();
    }
};

struct CullResult
{
    std::vector<int>        visited;
    std::vector<int>        boundsTested;
    std::vector<int>        candidates;
    std::vector<int>        wholeSubtrees;
    std::vector<int>        drawItems;
    std::map<int, Decision> decision;
};

struct Report
{
    std::vector<std::string> failures;

    void check (bool condition, const std::string& message)
    {
        if (!condition) failures.push_back (message);
    }

    bool ok () const { return failures.empty (); }
};

std::set<int>
idSet (std::initializer_list<int> ids)
{ return std::set<int> (ids); }

std::set<int>
vectorSet (const std::vector<int>& ids)
{ return std::set<int> (ids.begin (), ids.end ()); }

std::string
nameSet (const Scene& scene, const std::set<int>& ids)
{
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (int id: ids)
    {
        if (!first) out << ", ";
        first = false;
        out << scene.name (id);
    }
    out << "}";
    return out.str ();
}

void
expectIdSet (
    Report&              report,
    const Scene&         scene,
    const std::string&   label,
    const std::set<int>& actual,
    const std::set<int>& expected)
{
    std::set<int> missing;
    std::set<int> unexpected;

    std::set_difference (
        expected.begin (),
        expected.end (),
        actual.begin (),
        actual.end (),
        std::inserter (missing, missing.end ()));
    std::set_difference (
        actual.begin (),
        actual.end (),
        expected.begin (),
        expected.end (),
        std::inserter (unexpected, unexpected.end ()));

    if (!missing.empty () || !unexpected.empty ())
    {
        std::ostringstream out;
        out << label << " mismatch";
        if (!missing.empty ())
            out << "; wrongly missing/disappeared: "
                << nameSet (scene, missing);
        if (!unexpected.empty ())
            out << "; wrongly retained/exempted: "
                << nameSet (scene, unexpected);
        report.check (false, out.str ());
    }
}

void
expectIdSet (
    Report&                 report,
    const Scene&            scene,
    const std::string&      label,
    const std::vector<int>& actual,
    const std::set<int>&    expected)
{ expectIdSet (report, scene, label, vectorSet (actual), expected); }

bool
sameResult (const CullResult& a, const CullResult& b)
{
    return a.visited == b.visited && a.boundsTested == b.boundsTested &&
           a.candidates == b.candidates && a.wholeSubtrees == b.wholeSubtrees &&
           a.drawItems == b.drawItems && a.decision == b.decision;
}

void
addChild (Scene& scene, int parent, int child)
{ scene.node (parent).children.push_back (child); }

void
makeGroup (
    Scene&                     scene,
    int                        id,
    const std::string&         name,
    const B3&                  box,
    std::initializer_list<int> children)
{
    Node& node = scene.node (id);
    node.name  = name;
    node.box   = box;
    node.children.assign (children);
}

void
makeBoxLeaf (
    Scene&             scene,
    int                id,
    const std::string& name,
    const B3&          box,
    const V3&          sample,
    bool               conservative = false)
{
    Node& node        = scene.node (id);
    node.name         = name;
    node.box          = box;
    node.usesSphere   = false;
    node.conservative = conservative;
    node.samples.push_back (sample);
}

void
makeSphereLeaf (
    Scene&             scene,
    int                id,
    const std::string& name,
    const S3&          sphere,
    const V3&          sample,
    bool               conservative = false)
{
    Node& node        = scene.node (id);
    node.name         = name;
    node.sphere       = sphere;
    node.usesSphere   = true;
    node.conservative = conservative;
    node.samples.push_back (sample);
}

B3
box3 (float x0, float y0, float z0, float x1, float y1, float z1)
{ return B3 (V3 (x0, y0, z0), V3 (x1, y1, z1)); }

std::vector<V3>
boxCorners (const B3& box)
{
    std::vector<V3> corners;
    corners.reserve (8);
    for (int xi = 0; xi < 2; ++xi)
        for (int yi = 0; yi < 2; ++yi)
            for (int zi = 0; zi < 2; ++zi)
                corners.push_back (
                    V3 (xi ? box.max.x : box.min.x,
                        yi ? box.max.y : box.min.y,
                        zi ? box.max.z : box.min.z));
    return corners;
}

// Independent geometric oracle for the two explicitly constructed cameras.
// It deliberately uses hand-derived clip volumes rather than FrustumTest.
bool
handDerivedVisible (CameraKind camera, const V3& p)
{
    const float epsilon = 1.0e-4f;

    if (camera == PERSPECTIVE_FRAME)
    {
        // Perspective: near/far are 1/30 and the window is +/-1 at the
        // near plane, so at depth d = -z the visible half-width is d.
        float depth = -p.z;
        return depth > 1.0f + epsilon && depth < 30.0f - epsilon &&
               std::fabs (p.x) < depth - epsilon &&
               std::fabs (p.y) < depth - epsilon;
    }

    // Orthographic camera is translated to z=40 and rotated 180 degrees
    // around Y.  It therefore looks toward +z, with a constant +/-4 box.
    float depth = p.z - 40.0f;
    return depth > 1.0f + epsilon && depth < 30.0f - epsilon &&
           std::fabs (p.x) < 4.0f - epsilon && std::fabs (p.y) < 4.0f - epsilon;
}

bool
anySampleActuallyVisible (const Scene& scene, int id, CameraKind camera)
{
    const Node& node = scene.node (id);
    for (std::size_t i = 0; i < node.samples.size (); ++i)
    {
        if (handDerivedVisible (camera, node.samples[i])) return true;
    }
    return false;
}

std::set<int>
actualVisibleLeaves (const Scene& scene, CameraKind camera)
{
    std::set<int> visible;
    for (int id = 0; id < NODE_COUNT; ++id)
    {
        const Node& node = scene.node (id);
        if (node.children.empty () &&
            anySampleActuallyVisible (scene, id, camera))
        {
            visible.insert (id);
        }
    }
    return visible;
}

bool
boundIsVisible (const FrustumTestf& frustumTest, const Node& node)
{
    if (node.usesSphere) return frustumTest.isVisible (node.sphere);
    return frustumTest.isVisible (node.box);
}

bool
boundIsCompletelyInside (const FrustumTestf& frustumTest, const Node& node)
{
    if (node.usesSphere) return frustumTest.completelyContains (node.sphere);
    return frustumTest.completelyContains (node.box);
}

class HierarchicalCuller
{
public:
    explicit HierarchicalCuller (const Scene& scene) : _scene (&scene) {}

    void beginFrame (const FrustumTestf& frustumTest)
    {
        ++_frame;
        _frustumTest = frustumTest;
        _result      = CullResult ();
    }

    CullResult cullScene ()
    {
        _result = CullResult ();
        for (std::size_t i = 0; i < _scene->roots.size (); ++i)
            visit (_scene->roots[i]);
        return _result;
    }

    unsigned long frame () const { return _frame; }

private:
    void visit (int id)
    {
        const Node& node = _scene->node (id);

        _result.visited.push_back (id);
        _result.boundsTested.push_back (id);
        _result.decision[id] = UNSEEN;

        // Root/parent box rejection removes the complete subtree.
        if (!boundIsVisible (_frustumTest, node))
        {
            _result.decision[id] = CULLED_BY_PARENT_BOX;
            return;
        }

        // Independent complete containment is required before skipping all
        // descendant tests and sending the whole subtree to drawing.
        if (boundIsCompletelyInside (_frustumTest, node) &&
            !node.children.empty ())
        {
            _result.decision[id] = ACCEPT_WHOLE_SUBTREE;
            _result.wholeSubtrees.push_back (id);
            _result.drawItems.push_back (id);
            return;
        }

        if (node.children.empty ())
        {
            _result.decision[id] = DRAW_VISIBLE_LEAF;
            _result.candidates.push_back (id);
            _result.drawItems.push_back (id);
            return;
        }

        // Partial parent/root box overlap: children must be classified using
        // their own box or sphere bounds.
        _result.decision[id] = PARTIAL_EXPAND_CHILDREN;
        for (std::size_t i = 0; i < node.children.size (); ++i)
            visit (node.children[i]);
    }

    const Scene*  _scene;
    FrustumTestf  _frustumTest;
    CullResult    _result;
    unsigned long _frame = 0;
};

void
collectLeaves (const Scene& scene, int id, std::set<int>& leaves)
{
    const Node& node = scene.node (id);
    if (node.children.empty ())
    {
        leaves.insert (id);
        return;
    }

    for (std::size_t i = 0; i < node.children.size (); ++i)
        collectLeaves (scene, node.children[i], leaves);
}

std::set<int>
drawnLeaves (const Scene& scene, const CullResult& result)
{
    std::set<int> drawn;

    for (std::size_t i = 0; i < result.wholeSubtrees.size (); ++i)
        collectLeaves (scene, result.wholeSubtrees[i], drawn);

    drawn.insert (result.candidates.begin (), result.candidates.end ());
    return drawn;
}

std::set<int>
descendantIds (const Scene& scene, int root)
{
    std::set<int>    descendants;
    std::vector<int> pending = scene.node (root).children;

    while (!pending.empty ())
    {
        int id = pending.back ();
        pending.pop_back ();
        if (descendants.insert (id).second)
        {
            const std::vector<int>& children = scene.node (id).children;
            pending.insert (pending.end (), children.begin (), children.end ());
        }
    }
    return descendants;
}

struct FramePlan
{
    CameraKind    camera;
    std::set<int> partialParents;
    std::set<int> visibleLeaves;
    std::set<int> conservativeLeaves;
    std::set<int> outsideLeaves;
    std::set<int> wholeRoots;
    std::set<int> culledRoots;
    std::set<int> expectedCandidates;
};

void
verifyFrameGeometry (
    Report&             report,
    const Scene&        scene,
    const FrustumTestf& frustumTest,
    const FramePlan&    plan)
{
    for (int id: plan.visibleLeaves)
    {
        report.check (
            anySampleActuallyVisible (scene, id, plan.camera),
            scene.name (id) +
                " fixture must contain an actually visible point");
        report.check (
            boundIsVisible (frustumTest, scene.node (id)),
            scene.name (id) + " visible object's own bound must pass");
    }

    for (int id: plan.conservativeLeaves)
    {
        report.check (
            !anySampleActuallyVisible (scene, id, plan.camera),
            scene.name (id) + " conservative fixture's sample must be outside");
        report.check (
            boundIsVisible (frustumTest, scene.node (id)),
            scene.name (id) +
                " conservative bound is intentionally a false positive");
    }

    for (int id: plan.outsideLeaves)
    {
        report.check (
            !anySampleActuallyVisible (scene, id, plan.camera),
            scene.name (id) + " outside fixture's sample must be outside");
        report.check (
            !boundIsVisible (frustumTest, scene.node (id)),
            scene.name (id) + " outside object's own bound must be rejected");
    }

    for (int id: plan.partialParents)
    {
        const Node& node = scene.node (id);
        report.check (
            !node.usesSphere,
            scene.name (id) + " parent hierarchy bound must be a box");
        report.check (
            frustumTest.isVisible (node.box),
            scene.name (id) + " partial parent box must overlap the frustum");
        report.check (
            !frustumTest.completelyContains (node.box),
            scene.name (id) +
                " partial parent box must not be fully contained");

        int             outsideCorners = 0;
        std::vector<V3> corners        = boxCorners (node.box);
        for (std::size_t i = 0; i < corners.size (); ++i)
            if (!handDerivedVisible (plan.camera, corners[i])) ++outsideCorners;
        report.check (
            outsideCorners > 0,
            scene.name (id) +
                " partial-overlap fixture needs an independently outside corner");
    }

    for (int id: plan.wholeRoots)
    {
        const Node& node = scene.node (id);
        report.check (
            frustumTest.completelyContains (node.box),
            scene.name (id) +
                " whole-subtree fast path needs complete box containment");

        bool            allCornersInside = true;
        std::vector<V3> corners          = boxCorners (node.box);
        for (std::size_t i = 0; i < corners.size (); ++i)
            allCornersInside = allCornersInside &&
                               handDerivedVisible (plan.camera, corners[i]);
        report.check (
            allCornersInside,
            scene.name (id) +
                " whole-subtree containment needs independent corner evidence");
    }
}

void
verifyFrameTraversal (
    Report&           report,
    const Scene&      scene,
    const FramePlan&  plan,
    const CullResult& result)
{
    expectIdSet (
        report,
        scene,
        "whole-subtree roots",
        result.wholeSubtrees,
        plan.wholeRoots);
    expectIdSet (
        report,
        scene,
        "draw candidates",
        result.candidates,
        plan.expectedCandidates);

    std::set<int> expectedDrawn = actualVisibleLeaves (scene, plan.camera);
    expectedDrawn.insert (
        plan.conservativeLeaves.begin (), plan.conservativeLeaves.end ());
    expectIdSet (
        report,
        scene,
        "objects covered by draw list",
        drawnLeaves (scene, result),
        expectedDrawn);

    for (int id: plan.partialParents)
    {
        std::map<int, Decision>::const_iterator found =
            result.decision.find (id);
        report.check (
            found != result.decision.end () &&
                found->second == PARTIAL_EXPAND_CHILDREN,
            scene.name (id) +
                " must expand children after partial box overlap");

        const Node& node = scene.node (id);
        for (int child: node.children)
        {
            report.check (
                std::find (
                    result.boundsTested.begin (),
                    result.boundsTested.end (),
                    child) != result.boundsTested.end (),
                scene.name (child) +
                    " must be tested against its own sphere/box after partial parent");
        }
    }

    for (int id: plan.visibleLeaves)
    {
        report.check (
            result.decision.count (id) != 0 &&
                result.decision.find (id)->second == DRAW_VISIBLE_LEAF,
            scene.name (id) + " must remain a draw candidate");
    }

    for (int id: plan.conservativeLeaves)
    {
        report.check (
            result.decision.count (id) != 0 &&
                result.decision.find (id)->second == DRAW_VISIBLE_LEAF,
            scene.name (id) +
                " is an allowed conservative edge/corner candidate, not a false negative");
    }

    for (int id: plan.outsideLeaves)
    {
        report.check (
            result.decision.count (id) != 0 &&
                result.decision.find (id)->second == CULLED_BY_PARENT_BOX,
            scene.name (id) + " must be culled by its own bound");
        report.check (
            std::find (
                result.candidates.begin (), result.candidates.end (), id) ==
                result.candidates.end (),
            scene.name (id) + " must not be submitted as a candidate");
    }

    for (int root: plan.wholeRoots)
    {
        report.check (
            result.decision.count (root) != 0 &&
                result.decision.find (root)->second == ACCEPT_WHOLE_SUBTREE,
            scene.name (root) + " must use the whole-subtree draw path");

        std::set<int> descendants = descendantIds (scene, root);
        for (int descendant: descendants)
        {
            bool visited = std::find (
                               result.visited.begin (),
                               result.visited.end (),
                               descendant) != result.visited.end ();
            report.check (
                !visited,
                scene.name (descendant) +
                    " must not be individually traversed after verified complete containment");
        }
    }

    for (int root: plan.culledRoots)
    {
        report.check (
            result.decision.count (root) != 0 &&
                result.decision.find (root)->second == CULLED_BY_PARENT_BOX,
            scene.name (root) + " must be rejected by its current-frame box");

        std::set<int> descendants = descendantIds (scene, root);
        for (int descendant: descendants)
        {
            bool visited = std::find (
                               result.visited.begin (),
                               result.visited.end (),
                               descendant) != result.visited.end ();
            report.check (
                !visited,
                scene.name (descendant) +
                    " belongs to a culled subtree and must not be tested");
        }
    }
}

Scene
makeScene ()
{
    Scene scene;

    // Perspective-frame box hierarchy, centered around z=-10 where the
    // symmetric perspective frustum has x/y limits of about +/-10.
    makeGroup (
        scene,
        P_BOX_PARENT,
        "p.boxParent",
        box3 (-15.0f, -2.0f, -11.0f, -6.0f, 2.0f, -9.0f),
        {P_BOX_INSIDE, P_BOX_EDGE_VISIBLE, P_BOX_OUTSIDE, P_BOX_CONSERVATIVE});

    makeBoxLeaf (
        scene,
        P_BOX_INSIDE,
        "p.box.inside",
        box3 (-7.0f, -0.5f, -10.5f, -6.0f, 0.5f, -9.5f),
        V3 (-6.5f, 0.0f, -10.0f));
    makeBoxLeaf (
        scene,
        P_BOX_EDGE_VISIBLE,
        "p.box.edgeActuallyVisible",
        box3 (-10.6f, -0.5f, -10.5f, -8.8f, 0.5f, -9.5f),
        V3 (-9.7f, 0.0f, -10.0f));
    makeBoxLeaf (
        scene,
        P_BOX_OUTSIDE,
        "p.box.outside",
        box3 (-14.0f, -0.5f, -10.5f, -13.0f, 0.5f, -9.5f),
        V3 (-13.5f, 0.0f, -10.0f));
    makeBoxLeaf (
        scene,
        P_BOX_CONSERVATIVE,
        "p.box.conservativeCorner",
        box3 (-11.5f, -0.5f, -10.5f, -9.0f, 0.5f, -9.5f),
        V3 (-11.2f, 0.0f, -10.0f),
        true);

    // Perspective-frame sphere hierarchy.
    makeGroup (
        scene,
        P_SPHERE_PARENT,
        "p.sphereParent",
        box3 (5.0f, -2.0f, -11.0f, 15.0f, 2.0f, -9.0f),
        {P_SPHERE_INSIDE,
         P_SPHERE_EDGE_VISIBLE,
         P_SPHERE_OUTSIDE,
         P_SPHERE_CONSERVATIVE});

    makeSphereLeaf (
        scene,
        P_SPHERE_INSIDE,
        "p.sphere.inside",
        S3 (V3 (6.5f, 0.0f, -10.0f), 0.5f),
        V3 (6.5f, 0.0f, -10.0f));
    makeSphereLeaf (
        scene,
        P_SPHERE_EDGE_VISIBLE,
        "p.sphere.edgeActuallyVisible",
        S3 (V3 (9.6f, 0.0f, -10.0f), 1.0f),
        V3 (9.6f, 0.0f, -10.0f));
    makeSphereLeaf (
        scene,
        P_SPHERE_OUTSIDE,
        "p.sphere.outside",
        S3 (V3 (13.5f, 0.0f, -10.0f), 0.5f),
        V3 (13.5f, 0.0f, -10.0f));
    makeSphereLeaf (
        scene,
        P_SPHERE_CONSERVATIVE,
        "p.sphere.conservativeCorner",
        S3 (V3 (11.2f, 0.0f, -10.0f), 1.8f),
        V3 (11.2f, 0.0f, -10.0f),
        true);

    // Fully contained perspective subtree.
    makeGroup (
        scene,
        P_WHOLE_ROOT,
        "p.wholeRoot",
        box3 (-1.5f, -1.5f, -6.5f, 1.5f, 1.5f, -3.5f),
        {P_WHOLE_BRANCH, P_WHOLE_DIRECT_LEAF});
    makeGroup (
        scene,
        P_WHOLE_BRANCH,
        "p.wholeBranch",
        box3 (-1.2f, -1.0f, -6.0f, -0.2f, 1.0f, -4.0f),
        {P_WHOLE_NESTED_LEAF});
    makeBoxLeaf (
        scene,
        P_WHOLE_NESTED_LEAF,
        "p.whole.nestedLeaf",
        box3 (-1.0f, -0.4f, -5.4f, -0.4f, 0.4f, -4.6f),
        V3 (-0.7f, 0.0f, -5.0f));
    makeBoxLeaf (
        scene,
        P_WHOLE_DIRECT_LEAF,
        "p.whole.directLeaf",
        box3 (0.3f, -0.4f, -5.4f, 1.1f, 0.4f, -4.6f),
        V3 (0.7f, 0.0f, -5.0f));

    // Moved orthographic camera looks at z=50.  Its constant x/y range is
    // +/-4, so the same structural cases are rebuilt there in world coords.
    makeGroup (
        scene,
        O_BOX_PARENT,
        "o.boxParent",
        box3 (-8.0f, -2.0f, 49.0f, 0.0f, 2.0f, 51.0f),
        {O_BOX_INSIDE, O_BOX_EDGE_VISIBLE, O_BOX_OUTSIDE, O_BOX_CONSERVATIVE});

    makeBoxLeaf (
        scene,
        O_BOX_INSIDE,
        "o.box.inside",
        box3 (-2.7f, -0.5f, 49.5f, -1.3f, 0.5f, 50.5f),
        V3 (-2.0f, 0.0f, 50.0f));
    makeBoxLeaf (
        scene,
        O_BOX_EDGE_VISIBLE,
        "o.box.edgeActuallyVisible",
        box3 (-4.6f, -0.5f, 49.5f, -3.0f, 0.5f, 50.5f),
        V3 (-3.7f, 0.0f, 50.0f));
    makeBoxLeaf (
        scene,
        O_BOX_OUTSIDE,
        "o.box.outside",
        box3 (-7.0f, -0.5f, 49.5f, -6.0f, 0.5f, 50.5f),
        V3 (-6.5f, 0.0f, 50.0f));
    makeBoxLeaf (
        scene,
        O_BOX_CONSERVATIVE,
        "o.box.conservativeCorner",
        box3 (-5.2f, -0.4f, 49.6f, -3.4f, 0.4f, 50.4f),
        V3 (-4.8f, 0.0f, 50.0f),
        true);

    makeGroup (
        scene,
        O_SPHERE_PARENT,
        "o.sphereParent",
        box3 (0.0f, -2.0f, 49.0f, 8.0f, 2.0f, 51.0f),
        {O_SPHERE_INSIDE,
         O_SPHERE_EDGE_VISIBLE,
         O_SPHERE_OUTSIDE,
         O_SPHERE_CONSERVATIVE});

    makeSphereLeaf (
        scene,
        O_SPHERE_INSIDE,
        "o.sphere.inside",
        S3 (V3 (2.0f, 0.0f, 50.0f), 0.5f),
        V3 (2.0f, 0.0f, 50.0f));
    makeSphereLeaf (
        scene,
        O_SPHERE_EDGE_VISIBLE,
        "o.sphere.edgeActuallyVisible",
        S3 (V3 (3.7f, 0.0f, 50.0f), 0.8f),
        V3 (3.7f, 0.0f, 50.0f));
    makeSphereLeaf (
        scene,
        O_SPHERE_OUTSIDE,
        "o.sphere.outside",
        S3 (V3 (6.5f, 0.0f, 50.0f), 0.5f),
        V3 (6.5f, 0.0f, 50.0f));
    makeSphereLeaf (
        scene,
        O_SPHERE_CONSERVATIVE,
        "o.sphere.conservativeCorner",
        S3 (V3 (4.8f, 0.0f, 50.0f), 1.2f),
        V3 (4.8f, 0.0f, 50.0f),
        true);

    makeGroup (
        scene,
        O_WHOLE_ROOT,
        "o.wholeRoot",
        box3 (-1.5f, -1.5f, 49.0f, 1.5f, 1.5f, 51.0f),
        {O_WHOLE_BRANCH, O_WHOLE_DIRECT_LEAF});
    makeGroup (
        scene,
        O_WHOLE_BRANCH,
        "o.wholeBranch",
        box3 (-1.2f, -1.0f, 49.4f, -0.2f, 1.0f, 50.6f),
        {O_WHOLE_NESTED_LEAF});
    makeBoxLeaf (
        scene,
        O_WHOLE_NESTED_LEAF,
        "o.whole.nestedLeaf",
        box3 (-1.0f, -0.4f, 49.7f, -0.4f, 0.4f, 50.3f),
        V3 (-0.7f, 0.0f, 50.0f));
    makeBoxLeaf (
        scene,
        O_WHOLE_DIRECT_LEAF,
        "o.whole.directLeaf",
        box3 (0.3f, -0.4f, 49.7f, 1.1f, 0.4f, 50.3f),
        V3 (0.7f, 0.0f, 50.0f));

    scene.roots = {
        P_BOX_PARENT,
        P_SPHERE_PARENT,
        P_WHOLE_ROOT,
        O_BOX_PARENT,
        O_SPHERE_PARENT,
        O_WHOLE_ROOT};

    return scene;
}

FramePlan
perspectivePlan ()
{
    FramePlan plan;
    plan.camera         = PERSPECTIVE_FRAME;
    plan.partialParents = idSet ({P_BOX_PARENT, P_SPHERE_PARENT});
    plan.visibleLeaves  = idSet (
        {P_BOX_INSIDE,
         P_BOX_EDGE_VISIBLE,
         P_SPHERE_INSIDE,
         P_SPHERE_EDGE_VISIBLE});
    plan.conservativeLeaves =
        idSet ({P_BOX_CONSERVATIVE, P_SPHERE_CONSERVATIVE});
    plan.outsideLeaves = idSet ({P_BOX_OUTSIDE, P_SPHERE_OUTSIDE});
    plan.wholeRoots    = idSet ({P_WHOLE_ROOT});
    plan.culledRoots   = idSet ({O_BOX_PARENT, O_SPHERE_PARENT, O_WHOLE_ROOT});
    plan.expectedCandidates = idSet (
        {P_BOX_INSIDE,
         P_BOX_EDGE_VISIBLE,
         P_BOX_CONSERVATIVE,
         P_SPHERE_INSIDE,
         P_SPHERE_EDGE_VISIBLE,
         P_SPHERE_CONSERVATIVE});
    return plan;
}

FramePlan
orthographicPlan ()
{
    FramePlan plan;
    plan.camera         = MOVED_ORTHOGRAPHIC_FRAME;
    plan.partialParents = idSet ({O_BOX_PARENT, O_SPHERE_PARENT});
    plan.visibleLeaves  = idSet (
        {O_BOX_INSIDE,
         O_BOX_EDGE_VISIBLE,
         O_SPHERE_INSIDE,
         O_SPHERE_EDGE_VISIBLE});
    plan.conservativeLeaves =
        idSet ({O_BOX_CONSERVATIVE, O_SPHERE_CONSERVATIVE});
    plan.outsideLeaves = idSet ({O_BOX_OUTSIDE, O_SPHERE_OUTSIDE});
    plan.wholeRoots    = idSet ({O_WHOLE_ROOT});
    plan.culledRoots   = idSet ({P_BOX_PARENT, P_SPHERE_PARENT, P_WHOLE_ROOT});
    plan.expectedCandidates = idSet (
        {O_BOX_INSIDE,
         O_BOX_EDGE_VISIBLE,
         O_BOX_CONSERVATIVE,
         O_SPHERE_INSIDE,
         O_SPHERE_EDGE_VISIBLE,
         O_SPHERE_CONSERVATIVE});
    return plan;
}

} // namespace

void
testHierarchicalCulling ()
{
    std::cout << "Testing hierarchical frustum culling integration";

    Report report;
    Scene  scene = makeScene ();

    Frustumf perspectiveFrustum (1.0f, 30.0f, -1.0f, 1.0f, 1.0f, -1.0f, false);
    Matrix44<float> perspectiveCamera;
    perspectiveCamera.makeIdentity ();
    FrustumTestf perspectiveTest (perspectiveFrustum, perspectiveCamera);

    const float pi = 3.14159265358979323846f;
    Frustumf orthographicFrustum (1.0f, 30.0f, -4.0f, 4.0f, 4.0f, -4.0f, true);
    Matrix44<float> orthographicCamera;
    orthographicCamera.makeIdentity ();
    orthographicCamera.translate (V3 (0.0f, 0.0f, 40.0f));
    orthographicCamera.rotate (V3 (0.0f, pi, 0.0f));
    FrustumTestf orthographicTest (orthographicFrustum, orthographicCamera);

    report.check (
        !perspectiveFrustum.orthographic (),
        "first frame must use a perspective projection");
    report.check (
        orthographicFrustum.orthographic (),
        "second frame must use an orthographic projection");
    report.check (
        handDerivedVisible (PERSPECTIVE_FRAME, V3 (-9.7f, 0.0f, -10.0f)),
        "perspective hand oracle sanity check");
    report.check (
        handDerivedVisible (MOVED_ORTHOGRAPHIC_FRAME, V3 (-3.7f, 0.0f, 50.0f)),
        "orthographic hand oracle sanity check");

    FramePlan pPlan = perspectivePlan ();
    FramePlan oPlan = orthographicPlan ();

    verifyFrameGeometry (report, scene, perspectiveTest, pPlan);
    verifyFrameGeometry (report, scene, orthographicTest, oPlan);

    HierarchicalCuller culler (scene);

    // Alternate frames on the same nodes and the same culler.  Each returned
    // snapshot must belong only to the camera/frame that produced it.
    culler.beginFrame (perspectiveTest);
    CullResult pFirst = culler.cullScene ();

    culler.beginFrame (orthographicTest);
    CullResult oFirst = culler.cullScene ();

    culler.beginFrame (perspectiveTest);
    CullResult pSecond = culler.cullScene ();

    culler.beginFrame (orthographicTest);
    CullResult oSecond = culler.cullScene ();

    report.check (
        sameResult (pFirst, pSecond),
        "perspective frame result changed after running the orthographic frame");
    report.check (
        sameResult (oFirst, oSecond),
        "orthographic frame result changed after running the perspective frame");
    report.check (
        !sameResult (pFirst, oFirst),
        "perspective and orthographic current-frame results must not be shared");

    verifyFrameTraversal (report, scene, pPlan, pFirst);
    verifyFrameTraversal (report, scene, oPlan, oFirst);
    verifyFrameTraversal (report, scene, pPlan, pSecond);
    verifyFrameTraversal (report, scene, oPlan, oSecond);

    if (!report.ok ())
    {
        std::cout << "\n";
        for (std::size_t i = 0; i < report.failures.size (); ++i)
            std::cout << "  hierarchical culling failure: "
                      << report.failures[i] << "\n";
        std::cout << std::flush;
        assert (report.ok ());
    }

    std::cout << " ... passed\n\n";
}
