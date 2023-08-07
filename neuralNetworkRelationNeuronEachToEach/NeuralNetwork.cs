using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace neuralNetworkRelationNeuronEachToEach
{
    public class NeuralNetwork
    {
        public LayerNeiron[] Layers;
        public int countInputNeirons;
        public int countHideLayers;
        public int countNeironInHideLayer;
        public int countOutputNeirons;
        public double smoothing;
        public decimal errorTolerance;
        private int idCounter = 0;
        public Task[] initTasks;

        public NeuralNetwork(int countInputNeirons, int[] hidenLayersList, int countOutputNeirons, double smoothing, decimal errorTolerance)
        {
            this.countInputNeirons = countInputNeirons;
            this.countOutputNeirons = countOutputNeirons;
            this.smoothing = smoothing;
            this.errorTolerance = errorTolerance;
            Layers = new LayerNeiron[2 + hidenLayersList.Length];
            initTasks = new Task[countInputNeirons+1 + hidenLayersList.Sum()+hidenLayersList.Length+ countOutputNeirons];
            CreateInputLayer();
            CreateHiddenLayers(hidenLayersList);
            CreateOutputLayer();
            Task.WaitAll(initTasks);
        }

        public NeuralNetwork()
        {
        }

        public void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < countInputNeirons; i++)
            {
                var neuron = new Neuron(idCounter, 1, -1, NeuronType.Input);
                initTasks[idCounter] = neuron.initTask;
                //initTasks[idCounter].Start();
                initTasks[idCounter].RunSynchronously();
                idCounter++;
                inputNeurons.Add(neuron);
            }
            inputNeurons.Add(new Neuron(idCounter, 1, 4, NeuronType.bies));
            initTasks[idCounter] = inputNeurons.Last().initTask;
            //initTasks[idCounter].Start();
            initTasks[idCounter].RunSynchronously();
            idCounter++;
            var inputLayer = new LayerNeiron(inputNeurons, NeuronType.Input);
            Layers[0] = inputLayer;
        }

        private void CreateHiddenLayers(int[] list)
        {
            for (int i = 0; i < list.Length; i++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers[i];
                for (int k = 0; k < list[i]; k++)
                {
                    var neuron = new Neuron(idCounter, lastLayer.NeuronCount, -1);
                    initTasks[idCounter] = neuron.initTask;
                    //initTasks[idCounter].Start();
                    initTasks[idCounter].RunSynchronously();
                    idCounter++;
                    hiddenNeurons.Add(neuron);
                }
                hiddenNeurons.Add(new Neuron(idCounter, lastLayer.NeuronCount, 4, NeuronType.bies));
                initTasks[idCounter] = hiddenNeurons.Last().initTask;
                //initTasks[idCounter].Start();
                initTasks[idCounter].RunSynchronously();
                idCounter++;
                var hiddenLayer = new LayerNeiron(hiddenNeurons);
                Layers[i + 1] = hiddenLayer;
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers[Layers.Length - 2];
            for (int i = 0; i < countOutputNeirons; i++)
            {
                var neuron = new Neuron(idCounter, lastLayer.NeuronCount, i, NeuronType.Output);
                initTasks[idCounter] = neuron.initTask;
                //initTasks[idCounter].Start();
                initTasks[idCounter].RunSynchronously();
                idCounter++;
                outputNeurons.Add(neuron);
            }
            var outputLayer = new LayerNeiron(outputNeurons, NeuronType.Output);
            Layers[Layers.Length - 1] = outputLayer;
        }

        public List<Neuron> Predict(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (countOutputNeirons == 1)
            {
                return Layers.Last().Neurons;
            }
            else
            {
                return Layers.Last().Neurons;
            }
        }

        public void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        public void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Length; i++)
            {
                var layer = Layers[i];
                var previousLayerSingals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSingals);
                }
           
            }
        }

        public double Learn(double[] expected, double[,] inputs)
        {
            var error = 90.0m;
            var lastError = 1.0m;
            ulong i = 0;
            do
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);
                    error = Convert.ToDecimal(Backpropagation(output, input));
                }
                i++;
                if (lastError == error)
                {
                    Console.WriteLine("rep");
                    return 0;
                }
                if (i % 1000 == 0)
                    Console.WriteLine($"iteration: {i},\t error - {error},\tlast error - {lastError}");

                lastError = error;
            }
            while ((error > Convert.ToDecimal(errorTolerance) || error < Convert.ToDecimal(-errorTolerance)));
            return 0;
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.GetUpperBound(0)+1; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error = Backpropagation(output, input);
                }
                if (i % 1000 == 0)
                    Console.WriteLine($"iteration: {i},\t error - {error},\t");
            }

            var result = error / epoch;
            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
        }
        
        public double Backpropagation(double exprected, params double[] inputs)
        {
            var actual = Predict(inputs);
            double[] differences = new double[Layers.Last().Neurons.Count];
            for (int i = 0; i < Layers.Last().Neurons.Count; i++)
            {
                double differenceCurrentNeuron = 0;
                if (Layers.Last().Neurons.Count == 1)
                {
                    differenceCurrentNeuron = actual[i].Output - exprected;
                }
                else
                {
                    differenceCurrentNeuron = i == exprected ? 1 - actual[i].Output : -actual[i].Output;
                }
                differences[i] = differenceCurrentNeuron;

                Layers.Last().Neurons[i].Learn(differenceCurrentNeuron, smoothing);
            }
            for (int j = Layers.Length - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        if (previousNeuron.NeuronType == NeuronType.bies)
                        {
                            continue;
                        }
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, smoothing, previousNeuron.Delta);
                    }
                }
                
            }
            double multiplyResult = 1;
            return differences.Sum(x=>x*x);
        }

        public void saveNN(string fileName)
        {
            string json = JsonConvert.SerializeObject(this);            
            System.IO.File.WriteAllText($"{fileName}.json", json);
        }

        public static NeuralNetwork loadNN(string filename)
        {
            var nn = new NeuralNetwork();
            string jsonAll = System.IO.File.ReadAllText($"{filename}.json");
            dynamic staff = JObject.Parse(jsonAll);
            nn= JsonConvert.DeserializeObject<NeuralNetwork>(staff.ToString());
            nn.Layers = JsonConvert.DeserializeObject<List<LayerNeiron>>(staff.Layers.ToString());
            return nn;
        }
    }
}
