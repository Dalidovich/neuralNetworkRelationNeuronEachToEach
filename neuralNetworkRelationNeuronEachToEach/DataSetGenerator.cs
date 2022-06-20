namespace neuralNetworkRelationNeuronEachToEach
{
    public class DataSetGenerator
    {
        static Random rnd = new Random();
        public static double[] getInputDataRowArray(int count, int maxVaalue = 10)
        {
            var input = new List<double>();
            for (int i = 0; i < count; i++)
            {
                input.Add(rnd.Next(maxVaalue));
            }
            return input.ToArray();
        }
        public static double getMaxId(double[] a)
        {
            int idMax = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] > a[idMax])
                {
                    idMax = i;
                }
            }
            return idMax + 1;
        }
        public static List<double[]> getInputList(int countDataInRow, int countDataInCollum, int maxValueData = 10)
        {
            var list = new List<double[]>();
            for (int i = 0; i < countDataInRow; i++)
            {
                list.Add(getInputDataRowArray(countDataInCollum, maxValueData));
            }
            return list;
        }
        public static List<double> getOutputList(List<double[]> input)
        {
            var list = new List<double>();
            foreach (var item in input)
            {
                list.Add(getMaxId(item));
            }
            return list;
        }
        public static double[,] getInputsMatrix(List<double[]> inputList)
        {
            var input = new double[inputList.Count, inputList[0].Length];
            for (int i = 0; i < inputList.Count; i++)
            {
                for (int k = 0; k < inputList[i].Length; k++)
                {
                    input[i, k] = inputList[i][k];
                }
            }
            return input;
        }
        public static void showInputOutputData(double[] expected, double[,] inputs)
        {
            for (int i = 0; i < inputs.GetUpperBound(0) + 1; i++)
            {
                Console.Write($"expected - {expected[i]}\t");
                for (int k = 0; k < inputs.GetUpperBound(1) + 1; k++)
                {
                    Console.Write($"inputs - {inputs[i, k]},");
                }
                Console.WriteLine();
            }
        }
    }
}
