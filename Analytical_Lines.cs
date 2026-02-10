using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using Rhino.Geometry;

namespace End_Point_Predictor
{
    public class Analytical_Lines : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the Analytical_Lines class.
        /// </summary>
        public Analytical_Lines()
          : base("Analytical_Lines (LocalBBox)", "AnaLineLBB",
              "Build an analytical axis line from Center+Direction, intersect with a LOCAL oriented bounding box (Direction = local X) to get endpoints.\n" +
              "Extend offsets the resulting endpoints along the axis (positive = extend, negative = shorten).",
              "Analytical Model", "Line maker")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Center", "C", "Center point(s) (tree)", GH_ParamAccess.tree);
            pManager.AddVectorParameter("Direction", "V", "Direction vector(s) (tree)", GH_ParamAccess.tree);

            // Geometry list indexed by branch index
            pManager.AddGenericParameter("Geometry", "G",
                "Geometry per beam (mesh/brep/pointcloud/curve). List index should match branch index.",
                GH_ParamAccess.list);

            pManager.AddNumberParameter("Extend", "E",
                "Endpoint offset along axis after intersection. +E extends both ends, -E shortens both ends.",
                GH_ParamAccess.item, 0.0);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddLineParameter("AxisLine", "L", "Final axis line (between endpoints after Extend) (tree)", GH_ParamAccess.tree);
            pManager.AddPointParameter("P1", "P1", "Endpoint 1 (tree)", GH_ParamAccess.tree);
            pManager.AddPointParameter("P2", "P2", "Endpoint 2 (tree)", GH_ParamAccess.tree);

            pManager.AddBoxParameter("BBoxLocal", "B", "LOCAL oriented bounding box used for intersection (tree)", GH_ParamAccess.tree);
            pManager.AddPointParameter("BBoxCorners", "BC", "8 corners of the LOCAL oriented bounding box (tree)", GH_ParamAccess.tree);

            pManager.AddTextParameter("Log", "Log", "Debug log", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            GH_Structure<GH_Point> cTree;
            GH_Structure<GH_Vector> vTree;
            var geos = new List<object>();
            double extend = 0.0;

            if (!DA.GetDataTree(0, out cTree)) return;
            if (!DA.GetDataTree(1, out vTree)) return;
            if (!DA.GetDataList(2, geos)) return;
            if (!DA.GetData(3, ref extend)) return;

            var linesTree = new GH_Structure<GH_Line>();
            var p1Tree = new GH_Structure<GH_Point>();
            var p2Tree = new GH_Structure<GH_Point>();
            var boxTree = new GH_Structure<GH_Box>();
            var cornersTree = new GH_Structure<GH_Point>();

            var log = new StringBuilder();

            var paths = cTree.Paths;
            int branchCount = paths.Count;

            if (vTree.Paths.Count != branchCount)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "Center tree and Direction tree have different number of branches. Matching by branch index where possible.");
            }

            for (int i = 0; i < branchCount; i++)
            {
                GH_Path path = paths[i];

                // Center: first item in branch
                var cBranch = cTree.get_Branch(path);
                if (cBranch == null || cBranch.Count == 0) { log.AppendLine($"Branch {i}: empty Center branch."); continue; }
                var cGhp = cBranch[0] as GH_Point;
                if (cGhp == null) { log.AppendLine($"Branch {i}: Center is not GH_Point."); continue; }
                Point3d center = cGhp.Value;

                // Direction: first item in branch (path match or fallback index)
                IList vBranch = vTree.get_Branch(path);
                if (vBranch == null || vBranch.Count == 0)
                {
                    if (i < vTree.Paths.Count)
                        vBranch = vTree.get_Branch(vTree.Paths[i]);
                }
                if (vBranch == null || vBranch.Count == 0) { log.AppendLine($"Branch {i}: empty Direction branch."); continue; }

                var vGhv = vBranch[0] as GH_Vector;
                if (vGhv == null) { log.AppendLine($"Branch {i}: Direction is not GH_Vector."); continue; }

                Vector3d v = vGhv.Value;
                if (!v.Unitize() || v.IsTiny())
                {
                    log.AppendLine($"Branch {i}: direction vector invalid/tiny.");
                    continue;
                }

                // Geometry by list index
                if (i >= geos.Count)
                {
                    log.AppendLine($"Branch {i}: no Geometry at index {i}.");
                    continue;
                }

                // Build LOCAL OBB from geometry
                if (!TryBuildLocalObbFromGeometry(geos[i], v, out Box obb, out double diagLocal, out string geoInfo))
                {
                    log.AppendLine($"Branch {i}: local bbox failed ({geoInfo}).");
                    continue;
                }

                // Intersect line with LOCAL box to get endpoints
                double halfLen = Math.Max(diagLocal * 2.0, 1.0);
                Line longLine = new Line(center - v * halfLen, center + v * halfLen);

                if (!IntersectLineBox(longLine, obb, out Point3d ip1Raw, out Point3d ip2Raw))
                {
                    log.AppendLine($"Branch {i}: line-box intersection failed.");
                    boxTree.Append(new GH_Box(obb), path);
                    AppendBoxCorners(obb, cornersTree, path);
                    continue;
                }

                // Ensure ordering along v
                Point3d p1 = ip1Raw;
                Point3d p2 = ip2Raw;

                double t1 = Vector3d.Multiply(p1 - center, v);
                double t2 = Vector3d.Multiply(p2 - center, v);
                if (t1 > t2)
                {
                    var tmpP = p1; p1 = p2; p2 = tmpP;
                    var tmpT = t1; t1 = t2; t2 = tmpT;
                }

                // Extend semantics: move endpoints AFTER intersection
                if (Math.Abs(extend) > 1e-12)
                {
                    Point3d p1Moved = p1 - v * extend;
                    Point3d p2Moved = p2 + v * extend;

                    double newT1 = Vector3d.Multiply(p1Moved - center, v);
                    double newT2 = Vector3d.Multiply(p2Moved - center, v);
                    if (newT1 <= newT2)
                    {
                        p1 = p1Moved;
                        p2 = p2Moved;
                    }
                    else
                    {
                        log.AppendLine($"Branch {i}: Extend={extend} would invert endpoints; keeping bbox endpoints.");
                    }
                }

                Line outLine = new Line(p1, p2);

                linesTree.Append(new GH_Line(outLine), path);
                p1Tree.Append(new GH_Point(p1), path);
                p2Tree.Append(new GH_Point(p2), path);

                boxTree.Append(new GH_Box(obb), path);
                AppendBoxCorners(obb, cornersTree, path);
            }

            DA.SetDataTree(0, linesTree);
            DA.SetDataTree(1, p1Tree);
            DA.SetDataTree(2, p2Tree);
            DA.SetDataTree(3, boxTree);
            DA.SetDataTree(4, cornersTree);
            DA.SetData(5, log.ToString());
        }

        // =========================================================
        // Python-prinsipp i C#:
        //   - to_brep / unwrap
        //   - make_plane_from_x
        //   - oriented_bbox via duplicate+transform+GetBoundingBox
        // =========================================================

        private static bool TryBuildLocalObbFromGeometry(object gIn, Vector3d dirX, out Box obb, out double diagLocal, out string info)
        {
            obb = Box.Unset;
            diagLocal = 0.0;
            info = gIn == null ? "null" : gIn.GetType().FullName;

            // Unwrap GH goo -> Rhino geometry if possible
            object g = gIn;
            if (g is IGH_Goo goo && goo is IGH_GeometricGoo geoGoo)
            {
                g = geoGoo.ScriptVariable();
                info = "IGH_GeometricGoo -> " + (g == null ? "null" : g.GetType().FullName);
            }

            // Convert to Brep if possible
            Brep brep = ToBrep(g);
            if (brep == null)
            {
                // Support Mesh/Curve/PointCloud directly
                if (!(g is Mesh) && !(g is Curve) && !(g is PointCloud))
                {
                    info = "unsupported geometry type (not Brep/Mesh/Curve/PointCloud): " + (g == null ? "null" : g.GetType().FullName);
                    return false;
                }
            }

            // World bbox center -> plane origin
            BoundingBox bbWorld;
            if (brep != null) bbWorld = brep.GetBoundingBox(true);
            else if (g is Mesh m) bbWorld = m.GetBoundingBox(true);
            else if (g is Curve crv) bbWorld = crv.GetBoundingBox(true);
            else if (g is PointCloud pc) bbWorld = pc.GetBoundingBox(true);
            else { info = "could not compute world bbox"; return false; }

            if (!bbWorld.IsValid) { info = "world bbox invalid"; return false; }

            Point3d c = bbWorld.Min + (bbWorld.Max - bbWorld.Min) * 0.5;
            Point3d origin = new Point3d(c.X, c.Y, c.Z);

            Plane local = MakePlaneFromLocalX(origin, dirX);
            if (!local.IsValid) { info = "invalid local plane"; return false; }

            Plane world = Plane.WorldXY;

            Transform toWorldXY = Transform.PlaneToPlane(local, world);
            Transform back = Transform.PlaneToPlane(world, local);

            BoundingBox bb2;

            if (brep != null)
            {
                Brep b2 = brep.DuplicateBrep();
                b2.Transform(toWorldXY);
                bb2 = b2.GetBoundingBox(true);
                info = "Brep->local AABB via transform";
            }
            else if (g is Mesh m)
            {
                var m2 = m.DuplicateMesh();
                m2.Transform(toWorldXY);
                bb2 = m2.GetBoundingBox(true);
                info = "Mesh->local AABB via transform";
            }
            else if (g is Curve crv)
            {
                var c2 = (Curve)crv.Duplicate();
                c2.Transform(toWorldXY);
                bb2 = c2.GetBoundingBox(true);
                info = "Curve->local AABB via transform";
            }
            else
            {
                var pc = (PointCloud)g;
                var pc2 = new PointCloud(pc);
                pc2.Transform(toWorldXY);
                bb2 = pc2.GetBoundingBox(true);
                info = "PointCloud->local AABB via transform";
            }

            if (!bb2.IsValid) { info = "local AABB invalid"; return false; }

            Box boxWorld = new Box(world, bb2);
            if (!boxWorld.IsValid) { info = "boxWorld invalid"; return false; }

            boxWorld.Transform(back);

            obb = boxWorld;
            diagLocal = bb2.Diagonal.Length;
            return obb.IsValid;
        }

        private static Brep ToBrep(object g)
        {
            if (g == null) return null;

            if (g is Brep b) return b;
            if (g is Extrusion ex) return ex.ToBrep();
            if (g is Surface s) return Brep.CreateFromSurface(s);
            if (g is Mesh m) return Brep.CreateFromMesh(m, true);

            var mi = g.GetType().GetMethod("ToBrep", Type.EmptyTypes);
            if (mi != null)
            {
                try
                {
                    var res = mi.Invoke(g, null);
                    return res as Brep;
                }
                catch { return null; }
            }

            return null;
        }

        private static Plane MakePlaneFromLocalX(Point3d origin, Vector3d xAxis)
        {
            var x = xAxis;
            x.Unitize();

            Vector3d helper = Vector3d.ZAxis;
            if (Math.Abs(Vector3d.Multiply(x, helper)) > 0.95)
                helper = Vector3d.XAxis;

            Vector3d y = Vector3d.CrossProduct(helper, x);
            if (y.IsZero)
            {
                helper = Vector3d.YAxis;
                y = Vector3d.CrossProduct(helper, x);
            }
            y.Unitize();

            return new Plane(origin, x, y);
        }

        private static void AppendBoxCorners(Box b, GH_Structure<GH_Point> cornersTree, GH_Path path)
        {
            var corners = b.GetCorners();
            for (int i = 0; i < corners.Length; i++)
                cornersTree.Append(new GH_Point(corners[i]), path);
        }

        private static bool IntersectLineBox(Line line, Box box, out Point3d p1, out Point3d p2)
        {
            p1 = Point3d.Unset;
            p2 = Point3d.Unset;

            if (!box.IsValid) return false;

            Plane pl = box.Plane;
            Transform toLocal = Transform.PlaneToPlane(pl, Plane.WorldXY);
            Transform toWorld = Transform.PlaneToPlane(Plane.WorldXY, pl);

            Point3d a = line.From;
            Point3d b = line.To;
            a.Transform(toLocal);
            b.Transform(toLocal);

            Interval ix = box.X;
            Interval iy = box.Y;
            Interval iz = box.Z;

            Vector3d d = b - a;

            double tmin = 0.0;
            double tmax = 1.0;

            if (!ClipAxis(ix.Min, ix.Max, a.X, d.X, ref tmin, ref tmax)) return false;
            if (!ClipAxis(iy.Min, iy.Max, a.Y, d.Y, ref tmin, ref tmax)) return false;
            if (!ClipAxis(iz.Min, iz.Max, a.Z, d.Z, ref tmin, ref tmax)) return false;

            Point3d ia = a + d * tmin;
            Point3d ib = a + d * tmax;

            ia.Transform(toWorld);
            ib.Transform(toWorld);

            p1 = ia;
            p2 = ib;
            return true;
        }

        private static bool ClipAxis(double min, double max, double a, double d, ref double tmin, ref double tmax)
        {
            const double eps = 1e-12;

            if (Math.Abs(d) < eps)
            {
                return (a >= min && a <= max);
            }

            double t1 = (min - a) / d;
            double t2 = (max - a) / d;
            if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }

            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;

            return tmin <= tmax;
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("1D85CD45-0548-4ECD-9C55-C1BD31BFBC66"); }
        }
    }
}
