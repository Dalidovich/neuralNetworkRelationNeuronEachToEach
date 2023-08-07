using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace neuralNetworkRelationNeuronEachToEach
{
    public static class randomForTest
    {
        public static Random rnd = new Random();
        public static void setSeed(int seed)
        {
            rnd=new Random(seed);
        }
        public static double v()
        {
            return rnd.NextDouble();
        }
    }
}
