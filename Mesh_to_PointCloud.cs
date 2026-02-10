using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using Rhino.Geometry;

namespace End_Point_Predictor
{
    public class Mesh_to_PointCloud : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the Mesh_to_PointCloud class.
        /// </summary>
        public Mesh_to_PointCloud()
          : base("Mesh_to_PointCloud", "MkPC",
              "Distributes random points on mesh faces proportional to face area.\nOutputs both PointCloud and Locations (so you don't need Point Cloud Attributes).",
              "Analytical Model", "Point Tools")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Meshes", "M", "Input meshes", GH_ParamAccess.list);
            pManager.AddIntegerParameter("NrPts", "N", "Number of points per mesh", GH_ParamAccess.item, 1024);
            pManager.AddIntegerParameter("Seed", "S", "Random seed (-1 = random)", GH_ParamAccess.item, -1);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("PtsPerFace", "F", "Points per face (tree, branch per mesh)", GH_ParamAccess.tree);
            pManager.AddGenericParameter("PointClouds", "PC", "PointCloud per mesh", GH_ParamAccess.list);

            // Actual points (Locations)
            pManager.AddPointParameter("Locations", "P", "Point locations (tree, branch per mesh)", GH_ParamAccess.tree);

            pManager.AddNumberParameter("TotalArea", "A", "Total area per mesh", GH_ParamAccess.list);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var meshes = new List<Mesh>();
            int nrPts = 1024;
            int seed = 2;

            if (!DA.GetDataList(0, meshes)) return;
            if (!DA.GetData(1, ref nrPts)) return;
            DA.GetData(2, ref seed);

            if (meshes == null || meshes.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No meshes provided.");
                return;
            }

            if (nrPts < 0) nrPts = 0;

            Random rand = (seed >= 0) ? new Random(seed) : new Random();

            var pcs = new List<PointCloud>(meshes.Count);
            var areas = new List<double>(meshes.Count);

            var countsTree = new GH_Structure<GH_Integer>();
            var locationsTree = new GH_Structure<GH_Point>();

            for (int meshIndex = 0; meshIndex < meshes.Count; meshIndex++)
            {
                var m = meshes[meshIndex];
                var path = new GH_Path(meshIndex);

                if (m == null)
                {
                    pcs.Add(new PointCloud());
                    areas.Add(0.0);
                    continue;
                }

                // Triangulate for safety (uses A,B,C)
                Mesh mesh = m.DuplicateMesh();
                if (mesh.Faces.QuadCount > 0)
                    mesh.Faces.ConvertQuadsToTriangles();

                double areaM = 0.0;
                var amp = AreaMassProperties.Compute(mesh);
                if (amp != null) areaM = amp.Area;
                areas.Add(areaM);

                var pc = new PointCloud();

                if (areaM <= 0.0 || mesh.Faces.Count == 0 || nrPts == 0)
                {
                    pcs.Add(pc);
                    continue;
                }

                var faceCounts = new List<int>(mesh.Faces.Count);

                // Raw distribution per face
                for (int fi = 0; fi < mesh.Faces.Count; fi++)
                {
                    MeshFace mf = mesh.Faces[fi];

                    var A = mesh.Vertices[mf.A];
                    var B = mesh.Vertices[mf.B];
                    var C = mesh.Vertices[mf.C];

                    Vector3d AB = new Vector3d(B.X - A.X, B.Y - A.Y, B.Z - A.Z);
                    Vector3d AC = new Vector3d(C.X - A.X, C.Y - A.Y, C.Z - A.Z);

                    Vector3d cross = Vector3d.CrossProduct(AB, AC);
                    double areaFace = 0.5 * cross.Length;

                    double ares = (areaFace / areaM) * nrPts;
                    int res = Convert.ToInt32(Math.Round(ares));
                    if (res < 0) res = 0;

                    faceCounts.Add(res);
                }

                // Adjust so sum == nrPts
                AdjustCounts(faceCounts, nrPts, rand);

                // Generate points
                for (int fi = 0; fi < mesh.Faces.Count; fi++)
                {
                    MeshFace mf = mesh.Faces[fi];
                    var A = mesh.Vertices[mf.A];
                    var B = mesh.Vertices[mf.B];
                    var C = mesh.Vertices[mf.C];

                    int count = faceCounts[fi];
                    countsTree.Append(new GH_Integer(count), path);

                    if (count <= 0) continue;

                    var pts = RandomPointsOnTriangle(A, B, C, count, rand);

                    // add to PointCloud + Locations tree
                    pc.AddRange(pts);
                    for (int k = 0; k < pts.Count; k++)
                        locationsTree.Append(new GH_Point(pts[k]), path);
                }

                pcs.Add(pc);
            }

            DA.SetDataTree(0, countsTree);
            DA.SetDataList(1, pcs);
            DA.SetDataTree(2, locationsTree);
            DA.SetDataList(3, areas);
        }

        private static void AdjustCounts(List<int> counts, int target, Random rand)
        {
            if (counts == null || counts.Count == 0) return;
            if (target < 0) target = 0;

            int total = 0;
            for (int i = 0; i < counts.Count; i++) total += counts[i];
            if (total == target) return;

            if (total < target)
            {
                int deficit = target - total;
                while (deficit > 0)
                {
                    int minCount = int.MaxValue;
                    for (int i = 0; i < counts.Count; i++)
                        if (counts[i] < minCount) minCount = counts[i];

                    var candidates = new List<int>();
                    for (int i = 0; i < counts.Count; i++)
                        if (counts[i] == minCount) candidates.Add(i);

                    if (candidates.Count == 0) break;

                    counts[candidates[rand.Next(candidates.Count)]]++;
                    deficit--;
                }
            }
            else
            {
                int excess = total - target;
                while (excess > 0)
                {
                    int maxCount = 0;
                    for (int i = 0; i < counts.Count; i++)
                        if (counts[i] > maxCount) maxCount = counts[i];

                    if (maxCount <= 0) break;

                    var candidates = new List<int>();
                    for (int i = 0; i < counts.Count; i++)
                        if (counts[i] == maxCount) candidates.Add(i);

                    if (candidates.Count == 0) break;

                    counts[candidates[rand.Next(candidates.Count)]]--;
                    excess--;
                }
            }
        }

        public static List<Point3d> RandomPointsOnTriangle(Point3f A, Point3f B, Point3f C, int count, Random rand)
        {
            var pts = new List<Point3d>(count);

            for (int i = 0; i < count; i++)
            {
                double r1 = rand.NextDouble();
                double r2 = rand.NextDouble();

                double sqrtR1 = Math.Sqrt(r1);
                double u = 1.0 - sqrtR1;
                double v = sqrtR1 * (1.0 - r2);
                double w = sqrtR1 * r2;

                double x = (u * A.X + v * B.X + w * C.X);
                double y = (u * A.Y + v * B.Y + w * C.Y);
                double z = (u * A.Z + v * B.Z + w * C.Z);

                pts.Add(new Point3d(x, y, z));
            }

            return pts;
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid => new Guid("9171EB9F-16E2-4605-9ADE-C006F195B208");
    }
}
