using System;
using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;

namespace End_Point_Predictor
{
    public class End_Point_PredictorInfo : GH_AssemblyInfo
    {
        public override string Name => "End Point Predictor";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => null;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "";

        public override Guid Id => new Guid("52d744f2-7339-4a54-b56d-d95449dffb4a");

        //Return a string identifying you or your company.
        public override string AuthorName => "";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "";

        //Return a string representing the version.  This returns the same version as the assembly.
        public override string AssemblyVersion => GetType().Assembly.GetName().Version.ToString();
    }
}