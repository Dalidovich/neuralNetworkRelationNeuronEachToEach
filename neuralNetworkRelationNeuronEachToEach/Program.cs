using System.Diagnostics;

namespace neuralNetworkRelationNeuronEachToEach
{
    class Program
    {
        static void testSelectMaxIdWithAmong2Number(bool load = false)
        {
            var nn = new NeuralNetwork(2, new int[] { 2 }, 2, 0.01, 0.01m);
            var outputsList = new double[100];
            double[,] inputs = DataSetGenerator.createInputMatrixForMaxId(ref outputsList, outputsList.Length, 2);
            DataSetGenerator.showInputOutputData(outputsList.ToArray(), inputs);

            nn.Learn(outputsList.ToArray(), inputs);
            var countRight = 0;
            for (int i = 0; i < 10; i++)
            {
                var inputsForPredict = DataSetGenerator.generateInputUniqueNumberDataRowArray(2);
                var result = nn.Predict(inputsForPredict);
                Console.WriteLine($"{String.Join(",", inputsForPredict)}");
                Console.WriteLine($"expected - {DataSetGenerator.getMaxId(inputsForPredict)}");
                Console.WriteLine($"actual - {result.Find(x => x.Output == result.Max((y => y.Output))).OutputId}\n");
                if (DataSetGenerator.getMaxId(inputsForPredict) == result.Find(x => x.Output == result.Max((y => y.Output))).OutputId)
                {
                    countRight++;
                }
                foreach (var item in result)
                {
                    Console.WriteLine($" id - {item.OutputId}. value - {item.Output}");
                }
            }
            Console.WriteLine($"count right = {countRight}");
            if (countRight == 10)
            {
                nn.saveNN("selectIndexMaxIdWith2Input");
            }
        }
        static void testTime()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var nn = new NeuralNetwork(2, new int[] { 2, 2 }, 2, 0.001, 0.001m);
            sw.Stop();
            Console.WriteLine($"create NN time \'{sw.ElapsedMilliseconds}\' milliseconds");
            var outputsList = new double[100];
            double[,] inputs = DataSetGenerator.createInputMatrixForMaxId(ref outputsList, outputsList.Length, 2);
            Console.WriteLine(sw.ElapsedMilliseconds.ToString());
            sw.Reset();
            sw.Start();
            nn.Learn(outputsList.ToArray(), inputs,1);
            sw.Stop();
            Console.WriteLine($"learnint NN time \'{sw.ElapsedMilliseconds}\' milliseconds with 1 iteration");
        }
        static void Main(string[] args)
        {
            testSelectMaxIdWithAmong2Number();
            //testTime();
            Console.WriteLine("end");
        }
    }
}