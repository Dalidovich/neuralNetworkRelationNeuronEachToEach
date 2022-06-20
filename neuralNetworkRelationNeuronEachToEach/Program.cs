using System.Diagnostics;

namespace neuralNetworkRelationNeuronEachToEach
{
    class Program
    {
        static void testSelectMaxIdWith2InputAndSaveAfterLerning(bool load = false)
        {
            var nn = new NeuralNetwork();
            if (load)
            {
                try
                {
                    nn = NeuralNetwork.loadNN("selectIndexMaxIdWith2Input");
                }
                catch
                {
                    Console.WriteLine("error at load NN");
                    nn = new NeuralNetwork(2, 2, 2, 2, 0.001, 0.001m);
                }
            }
            else
            {
                nn = new NeuralNetwork(2, 2, 2, 2, 0.001, 0.001m);
                Console.WriteLine("now");
            }
            Console.WriteLine($"{nn.countNeironInHideLayer}");
            var inputsList = DataSetGenerator.getInputList(100, 2, 10);
            var outputsList = DataSetGenerator.getOutputList(inputsList);
            double[,] inputs = DataSetGenerator.getInputsMatrix(inputsList);
            DataSetGenerator.showInputOutputData(outputsList.ToArray(), inputs);

            nn.Learn(outputsList.ToArray(), inputs);
            var countRight = 0;
            for (int i = 0; i < 10; i++)
            {
                var inputsForPredict = DataSetGenerator.getInputDataRowArray(2);
                var result = nn.Predict(inputsForPredict);
                Console.WriteLine($"{String.Join(",", inputsForPredict)}");
                Console.WriteLine($"expected - {DataSetGenerator.getMaxId(inputsForPredict)}");
                Console.WriteLine($"actual - {result.Find(x => x.Output == result.Max((y => y.Output))).OutputId}\n");
                if (DataSetGenerator.getMaxId(inputsForPredict) == result.Find(x => x.Output == result.Max((y => y.Output))).OutputId)
                {
                    countRight++;
                }
                //foreach(var item in result)
                //{
                //    Console.WriteLine($" id - {item.OutputId}. value - {item.Output}");
                //}
            }
            Console.WriteLine($"count right = {countRight}");
            if (countRight == 10)
            {
                nn.saveNN("selectIndexMaxIdWith2Input");
            }
        }
        static void testSelectMaxIdWith3InputAndSaveAfterLerning(bool load = false)
        {
            var nn = new NeuralNetwork();
            if (load)
            {
                try
                {
                    nn = NeuralNetwork.loadNN("selectIndexMaxIdWith3Input");
                }
                catch
                {
                    Console.WriteLine("error at load NN");
                }
            }
            else
            {
                nn = new NeuralNetwork(3, 4, 5, 3, 0.001, 0.001m);
                Console.WriteLine("now");
            }
            Console.WriteLine($"{nn.countNeironInHideLayer}");
            var inputsList = DataSetGenerator.getInputList(100, 3, 10);
            var outputsList = DataSetGenerator.getOutputList(inputsList);
            double[,] inputs = DataSetGenerator.getInputsMatrix(inputsList);
            DataSetGenerator.showInputOutputData(outputsList.ToArray(), inputs);

            nn.Learn(outputsList.ToArray(), inputs);
            var countRight = 0;
            for (int i = 0; i < 10; i++)
            {
                var inputsForPredict = DataSetGenerator.getInputDataRowArray(3);
                var result = nn.Predict(inputsForPredict);
                Console.WriteLine($"{String.Join(",", inputsForPredict)}");
                Console.WriteLine($"expected - {DataSetGenerator.getMaxId(inputsForPredict)}");
                Console.WriteLine($"actual - {result.Find(x => x.Output == result.Max((y => y.Output))).OutputId}\n");
                if (DataSetGenerator.getMaxId(inputsForPredict) == result.Find(x => x.Output == result.Max((y => y.Output))).OutputId)
                {
                    countRight++;
                }
                //foreach(var item in result)
                //{
                //    Console.WriteLine($" id - {item.OutputId}. value - {item.Output}");
                //}
            }
            Console.WriteLine($"count right = {countRight}");
            if (countRight == 10)
            {
                nn.saveNN("selectIndexMaxIdWith3Input");
            }
        }
        static void testTime()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            var nn = new NeuralNetwork(2, 5, 5, 2, 0.001, 0.001m);
            sw.Stop();
            Console.WriteLine($"create NN time \'{sw.ElapsedMilliseconds}\' milliseconds");
            var inputsList = DataSetGenerator.getInputList(100, 2, 10);
            var outputsList = DataSetGenerator.getOutputList(inputsList);
            double[,] inputs = DataSetGenerator.getInputsMatrix(inputsList);
            Console.WriteLine(sw.ElapsedMilliseconds.ToString());
            sw.Reset();
            sw.Start();
            nn.Learn(outputsList.ToArray(), inputs, 1);
            sw.Stop();
            Console.WriteLine($"learnint NN time \'{sw.ElapsedMilliseconds}\' milliseconds with 1 iteration");
        }
        static void Main(string[] args)
        {
            //testSelectMaxIdWith3InputAndSaveAfterLerning();
            testSelectMaxIdWith2InputAndSaveAfterLerning(true);
            //testTime();
            Console.WriteLine("end");
            Console.Read();
        }
    }
}