using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using Rhino.Geometry;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace End_Point_Predictor
{
    public class End_Point_PredictorComponent : GH_Component
    {
        private readonly string defaultPythonExe = @"C:\Users\lotoppe\pt-cu121\Scripts\python.exe";

        public End_Point_PredictorComponent()
          : base("Predict Axis (Scaled+Centered)", "EPP_SCL",
              "Moves each pointcloud to origin, scales along fitted axis, runs ML in batch, then transforms c/v back.\n" +
              "Optionally uses Brep centroid (volume/area) instead of pointcloud centroid when Breps list is provided.\n" +
              "Breps are matched by index: beam i -> breps[i].",
              "Analytical Model", "Predictor")
        {
        }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            // 0
            pManager.AddPointParameter("PointsTree", "P", "Point clouds as DataTree (one branch per beam)", GH_ParamAccess.tree);

            // 1 ? NYTT: liste med breps (ikke tree)
            pManager.AddBrepParameter("Breps", "B",
                "Optional list of Breps. Matched by index: beam i uses breps[i]. If missing, falls back to pointcloud centroid.",
                GH_ParamAccess.list);
            pManager[pManager.ParamCount - 1].Optional = true;

            // 2
            pManager.AddNumberParameter("Scale", "s", "Scale factor along fitted axis (1 = no scaling)", GH_ParamAccess.item, 1.0);
            // 3
            pManager.AddTextParameter("ModelPath", "M", "Full path to trained model/checkpoint", GH_ParamAccess.item);
            // 4
            pManager.AddTextParameter("ScriptPath", "S", "Full path to Python script (.py) (batch text IO)", GH_ParamAccess.item);

            // 5 optional override
            pManager.AddTextParameter("PythonExe", "Py",
                "Optional full path to python.exe. If empty/not connected, uses default path embedded in the plugin.",
                GH_ParamAccess.item);
            pManager[pManager.ParamCount - 1].Optional = true;

            // 6
            pManager.AddBooleanParameter("Run", "R", "Run once on rising edge (use Button recommended)", GH_ParamAccess.item, false);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Center", "C", "Predicted center point in ORIGINAL coordinates (tree)", GH_ParamAccess.tree);
            pManager.AddVectorParameter("Direction", "U", "Predicted direction in ORIGINAL coordinates (tree)", GH_ParamAccess.tree);

            pManager.AddPointParameter("Centroid_Used", "O", "Centroid used to move each beam to origin (world coords, tree)", GH_ParamAccess.tree);
            pManager.AddPointParameter("Centroid_PC", "OP", "Pointcloud centroid (mean of points) (world coords, tree)", GH_ParamAccess.tree);
            pManager.AddPointParameter("Centroid_Brep", "OB", "Brep centroid (volume if closed else area) (world coords, tree)", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Centroid_Delta", "dO", "Distance |Centroid_PC - Centroid_Brep| per beam (world units)", GH_ParamAccess.tree);

            pManager.AddTextParameter("Log", "L", "stdout+stderr from Python + centroid diagnostics", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            GH_Structure<GH_Point> ptsTree;
            if (!DA.GetDataTree(0, out ptsTree)) return;

            // ? breps som LISTE
            var breps = new List<Brep>();
            bool hasBreps = DA.GetDataList(1, breps); // optional (kan være false hvis ikke koblet)

            double s = 1.0;
            string modelPath = null;
            string scriptPath = null;
            string userPythonExe = null;
            bool run = false;

            if (!DA.GetData(2, ref s)) return;
            if (!DA.GetData(3, ref modelPath)) return;
            if (!DA.GetData(4, ref scriptPath)) return;
            DA.GetData(5, ref userPythonExe);
            if (!DA.GetData(6, ref run)) return;

            if (!run) return;

            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "ModelPath is invalid or file does not exist.");
                return;
            }
            if (string.IsNullOrWhiteSpace(scriptPath) || !File.Exists(scriptPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Python script not found: " + scriptPath);
                return;
            }

            string pythonExe = string.IsNullOrWhiteSpace(userPythonExe) ? defaultPythonExe : userPythonExe;
            if (string.IsNullOrWhiteSpace(pythonExe) || !File.Exists(pythonExe))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "python.exe not found. Tried: " + pythonExe);
                return;
            }

            var ci = CultureInfo.InvariantCulture;

            string tempDir = Path.GetTempPath();
            string tag = this.InstanceGuid.ToString("N");
            string inPath = Path.Combine(tempDir, "gh_ml_batch_in_scl_" + tag + ".txt");
            string outPath = Path.Combine(tempDir, "gh_ml_batch_out_scl_" + tag + ".txt");

            var paths = ptsTree.Paths;

            var byId = new Dictionary<int, BeamXforms>();

            int beamCount = 0;

            // output trees for centroid debugging
            var centroidUsedTree = new GH_Structure<GH_Point>();
            var centroidPcTree = new GH_Structure<GH_Point>();
            var centroidBrepTreeOut = new GH_Structure<GH_Point>();
            var centroidDeltaTree = new GH_Structure<GH_Number>();

            var diagLog = new System.Text.StringBuilder();

            using (var sw = new StreamWriter(inPath))
            {
                for (int i = 0; i < paths.Count; i++)
                {
                    GH_Path path = paths[i];
                    IList branch = ptsTree.get_Branch(path);
                    if (branch == null || branch.Count == 0) continue;

                    var pts = new List<Point3d>(branch.Count);
                    for (int j = 0; j < branch.Count; j++)
                    {
                        var ghp = branch[j] as GH_Point;
                        if (ghp == null) continue;
                        pts.Add(ghp.Value);
                    }
                    if (pts.Count < 2) continue;

                    // centroid fra pointcloud
                    Point3d centroidPC = ComputeCentroid(pts);

                    // ? hent brep ved indeks i (hvis finnes)
                    bool brepCentroidOk = false;
                    Point3d centroidBrep = Point3d.Unset;

                    if (hasBreps && breps != null && i < breps.Count && breps[i] != null)
                    {
                        brepCentroidOk = TryComputeBrepCentroid(breps[i], out centroidBrep);
                    }

                    Point3d centroidUsed = brepCentroidOk ? centroidBrep : centroidPC;

                    double delta = double.NaN;
                    if (brepCentroidOk)
                        delta = centroidPC.DistanceTo(centroidBrep);

                    centroidUsedTree.Append(new GH_Point(centroidUsed), path);
                    centroidPcTree.Append(new GH_Point(centroidPC), path);
                    centroidBrepTreeOut.Append(new GH_Point(brepCentroidOk ? centroidBrep : Point3d.Unset), path);
                    centroidDeltaTree.Append(new GH_Number(delta), path);

                    diagLog.AppendLine(
                        string.Format(ci,
                            "beam_index {0} | path {1} | used=({2:F3},{3:F3},{4:F3}) | pc=({5:F3},{6:F3},{7:F3}) | brep_ok={8} | delta={9}",
                            i,
                            path.ToString(),
                            centroidUsed.X, centroidUsed.Y, centroidUsed.Z,
                            centroidPC.X, centroidPC.Y, centroidPC.Z,
                            brepCentroidOk ? "1" : "0",
                            double.IsNaN(delta) ? "nan" : delta.ToString("F6", ci)
                        )
                    );

                    // axis fra punkter (behold som før)
                    Vector3d axis = FitAxisFromPoints(pts);
                    if (!axis.Unitize())
                        axis = Vector3d.ZAxis;

                    // move centroidUsed -> origin
                    Vector3d moveVec = new Vector3d(-centroidUsed.X, -centroidUsed.Y, -centroidUsed.Z);
                    Transform xMove = Transform.Translation(moveVec);

                    // scale along axis
                    Transform xScale = Transform.Identity;

                    Vector3d zAxis = axis;
                    Vector3d up = Vector3d.ZAxis;
                    double dot = Vector3d.Multiply(up, zAxis);
                    if (Math.Abs(dot) > 0.99)
                        up = Vector3d.YAxis;

                    Vector3d xAxis = Vector3d.CrossProduct(up, zAxis);
                    if (!xAxis.Unitize())
                        xAxis = Vector3d.XAxis;

                    Vector3d yAxis = Vector3d.CrossProduct(zAxis, xAxis);
                    yAxis.Unitize();

                    Plane plane = new Plane(Point3d.Origin, xAxis, yAxis);

                    bool doScale = (s != 1.0) && (!double.IsNaN(s)) && (!double.IsInfinity(s));
                    if (doScale)
                        xScale = Transform.Scale(plane, 1.0, 1.0, s);

                    Transform xScaleInv = Transform.Identity;
                    Transform xMoveInv = Transform.Translation(-moveVec);

                    if (doScale)
                    {
                        Transform tmp = xScale;
                        if (!tmp.TryGetInverse(out xScaleInv))
                        {
                            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Could not invert scale transform for branch {i}.");
                            return;
                        }
                    }

                    byId[i] = new BeamXforms
                    {
                        Path = path,
                        Move = xMove,
                        MoveInv = xMoveInv,
                        Scale = xScale,
                        ScaleInv = xScaleInv,
                        DoScale = doScale
                    };

                    sw.WriteLine(string.Format(ci, "beam {0} {1}", i, path.ToString()));

                    for (int j = 0; j < pts.Count; j++)
                    {
                        Point3d q = new Point3d(pts[j]);
                        q.Transform(xMove);
                        if (doScale) q.Transform(xScale);

                        sw.WriteLine(string.Format(ci, "{0} {1} {2}", q.X, q.Y, q.Z));
                    }

                    sw.WriteLine("endbeam");
                    beamCount++;
                }
            }

            if (beamCount == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid branches found (need >=2 points per branch).");
                return;
            }

            string args = string.Format(ci,
                "\"{0}\" --input \"{1}\" --output \"{2}\" --checkpoint \"{3}\"",
                scriptPath, inPath, outPath, modelPath);

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            string stdOut = "";
            string stdErr = "";

            try
            {
                using (var proc = Process.Start(psi))
                {
                    stdOut = proc.StandardOutput.ReadToEnd();
                    stdErr = proc.StandardError.ReadToEnd();
                    proc.WaitForExit();

                    DA.SetData(6, stdOut + (string.IsNullOrWhiteSpace(stdErr) ? "" : ("\n" + stdErr)) +
                                   "\n\n--- Centroid diagnostics ---\n" + diagLog.ToString());

                    if (proc.ExitCode != 0)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Python process failed:\n" + stdErr);
                        return;
                    }
                }
            }
            catch (Exception ex)
            {
                DA.SetData(6, stdOut + "\n" + stdErr + "\n\n--- Centroid diagnostics ---\n" + diagLog.ToString());
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Error starting python process: " + ex.Message);
                return;
            }

            if (!File.Exists(outPath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Output file not found: " + outPath);
                return;
            }

            var results = ParseBatchOutput(outPath);

            var centerOrigTree = new GH_Structure<GH_Point>();
            var dirOrigTree = new GH_Structure<GH_Vector>();

            foreach (var kv in byId)
            {
                int id = kv.Key;
                BeamXforms xf = kv.Value;

                if (!results.ContainsKey(id)) continue;

                var r = results[id];

                Point3d cScaled = r.c;
                Vector3d vScaled = r.v;
                vScaled.Unitize();

                Point3d cOrig = new Point3d(cScaled);
                if (xf.DoScale) cOrig.Transform(xf.ScaleInv);
                cOrig.Transform(xf.MoveInv);

                Vector3d vOrig = new Vector3d(vScaled);
                if (xf.DoScale) vOrig.Transform(xf.ScaleInv);
                vOrig.Unitize();

                centerOrigTree.Append(new GH_Point(cOrig), xf.Path);
                dirOrigTree.Append(new GH_Vector(vOrig), xf.Path);
            }

            DA.SetDataTree(0, centerOrigTree);
            DA.SetDataTree(1, dirOrigTree);

            // centroid outputs
            DA.SetDataTree(2, centroidUsedTree);
            DA.SetDataTree(3, centroidPcTree);
            DA.SetDataTree(4, centroidBrepTreeOut);
            DA.SetDataTree(5, centroidDeltaTree);
        }

        // ----------------- Helpers -----------------

        private class BeamXforms
        {
            public GH_Path Path;
            public Transform Move;
            public Transform MoveInv;
            public Transform Scale;
            public Transform ScaleInv;
            public bool DoScale;
        }

        private class BeamResult
        {
            public Point3d c;
            public Vector3d v;
        }

        private static bool TryComputeBrepCentroid(Brep brep, out Point3d centroid)
        {
            centroid = Point3d.Unset;
            if (brep == null) return false;

            try
            {
                if (brep.IsSolid)
                {
                    var vmp = VolumeMassProperties.Compute(brep);
                    if (vmp != null)
                    {
                        centroid = vmp.Centroid;
                        return centroid.IsValid;
                    }
                }

                var amp = AreaMassProperties.Compute(brep);
                if (amp != null)
                {
                    centroid = amp.Centroid;
                    return centroid.IsValid;
                }
            }
            catch { }

            return false;
        }

        private Dictionary<int, BeamResult> ParseBatchOutput(string outPath)
        {
            var ci = CultureInfo.InvariantCulture;
            var dict = new Dictionary<int, BeamResult>();

            string[] lines = File.ReadAllLines(outPath);
            int idx = 0;

            while (idx < lines.Length)
            {
                string line = lines[idx].Trim();
                if (line.Length == 0) { idx++; continue; }

                if (!line.StartsWith("result "))
                {
                    idx++;
                    continue;
                }

                var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) throw new Exception("Bad result header: " + line);
                int id = int.Parse(parts[1], ci);

                Point3d c = Point3d.Unset;
                Vector3d v = Vector3d.Unset;

                idx++;
                while (idx < lines.Length)
                {
                    string l = lines[idx].Trim();
                    idx++;

                    if (l == "endresult") break;
                    if (l.Length == 0) continue;

                    var p = l.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length < 1) continue;

                    if (p[0] == "c")
                        c = ParsePointFromParts(p, 1, ci, l);
                    else if (p[0] == "v")
                        v = ParseVectorFromParts(p, 1, ci, l);
                }

                if (c != Point3d.Unset && v != Vector3d.Unset)
                    dict[id] = new BeamResult { c = c, v = v };
            }

            return dict;
        }

        private static Point3d ParsePointFromParts(string[] parts, int start, CultureInfo ci, string fullLine)
        {
            if (parts.Length < start + 3) throw new Exception("Cannot parse point from: " + fullLine);
            double x = double.Parse(parts[start + 0], ci);
            double y = double.Parse(parts[start + 1], ci);
            double z = double.Parse(parts[start + 2], ci);
            return new Point3d(x, y, z);
        }

        private static Vector3d ParseVectorFromParts(string[] parts, int start, CultureInfo ci, string fullLine)
        {
            if (parts.Length < start + 3) throw new Exception("Cannot parse vector from: " + fullLine);
            double x = double.Parse(parts[start + 0], ci);
            double y = double.Parse(parts[start + 1], ci);
            double z = double.Parse(parts[start + 2], ci);
            return new Vector3d(x, y, z);
        }

        private static Point3d ComputeCentroid(List<Point3d> pts)
        {
            double sx = 0, sy = 0, sz = 0;
            int n = pts.Count;
            for (int i = 0; i < n; i++)
            {
                sx += pts[i].X;
                sy += pts[i].Y;
                sz += pts[i].Z;
            }
            return new Point3d(sx / n, sy / n, sz / n);
        }

        private static Vector3d FitAxisFromPoints(List<Point3d> pts)
        {
            Line line;
            bool ok = Line.TryFitLineToPoints(pts, out line);
            if (!ok || line.Length <= 1e-9)
                return Vector3d.ZAxis;

            Vector3d axis = line.Direction;
            if (!axis.Unitize()) axis = Vector3d.ZAxis;
            return axis;
        }

        protected override System.Drawing.Bitmap Icon => null;

        public override Guid ComponentGuid => new Guid("ae3d4221-c835-450b-b079-65c857bc28ae");
    }
}
