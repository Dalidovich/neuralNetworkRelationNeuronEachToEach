using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace neuralNetworkRelationNeuronEachToEach
{
    public class NeuralNetwork
    {
        public List<LayerNeiron> Layers { get; private set;}
        public int countInputNeirons;
        public int countHideLayers;
        public int countNeironInHideLayer;
        public int countOutputNeirons;
        public double smoothing;
        public decimal errorTolerance;

        public NeuralNetwork(int countInputNeirons, int countHideLayers, int countCountNeironInHideLayer, int countOutputNeirons, double smoothing,decimal errorTolerance)
        {
            this.countInputNeirons = countInputNeirons;
            this.countHideLayers = countHideLayers;
            this.countNeironInHideLayer = countCountNeironInHideLayer;
            this.countOutputNeirons = countOutputNeirons;
            this.smoothing = smoothing;
            this.errorTolerance = errorTolerance;
            Layers = new List<LayerNeiron>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public NeuralNetwork()
        {
        }

        public void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < countInputNeirons; i++)
            {
                var neuron = new Neuron(1,-1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            inputNeurons.Add(new Neuron(1, 4, NeuronType.bies));
            var inputLayer = new LayerNeiron(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        private void CreateHiddenLayers()
        {
            for (int j = 0; j < countHideLayers; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < countNeironInHideLayer; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount,-1);
                    hiddenNeurons.Add(neuron);
                }
                hiddenNeurons.Add(new Neuron(lastLayer.NeuronCount, 4, NeuronType.bies));
                var hiddenLayer = new LayerNeiron(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }
        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < countOutputNeirons; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount,i+1, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new LayerNeiron(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
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
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }
        public void showLastLayer()
        {
            var last = Layers.Last();
            foreach (var item in last.Neurons)
            {
                Console.WriteLine($"id - {item.OutputId} with value {item.Output}, type - {item.NeuronType}, w - {item.Weights[0]}");
            }
            showHideLayer();
        }
        public void showHideLayer()
        {
            var last = Layers[1];
            foreach (var item in last.Neurons)
            {
                Console.WriteLine($"id - {item.OutputId} with value {item.Output}, type - {item.NeuronType}, w - {item.Weights[0]}");
            }
        }
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
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
                //Console.WriteLine("sd");
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);
                    i++;

                    error = Convert.ToDecimal(Backpropagation(output, input));
                }
                if (lastError == error)
                {
                    Console.WriteLine("rep");
                    return 0;
                }
                if (i % 10000 == 0)
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
                for (int j = 0; j < expected.GetUpperBound(0); j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    //Console.WriteLine("error - " + Convert.ToDecimal(error));
                    error = Backpropagation(output, input);
                }
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
        private double Backpropagation(double exprected, params double[] inputs)
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
                    differenceCurrentNeuron = i == exprected - 1 ? actual[i].Output - 1 : actual[i].Output;
                }
                differences[i] = differenceCurrentNeuron;
                Layers.Last().Neurons[i].Learn(differenceCurrentNeuron, smoothing);
            }            
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        //new Task(() => {
                            neuron.Learn(error, smoothing);
                        //}).Start();
                    }
                }
                
            }
            double multiplyResult = 1;
            differences.ToList().ForEach(x => multiplyResult = x * multiplyResult);
            return multiplyResult;
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
            //Console.WriteLine(staff.Layers[1].Type);
            //List<LayerNeiron> lln = new List<LayerNeiron>();
            //for (int i = 0; i < staff.Layers.Count; i++)
            //{
            //    LayerNeiron layer = new LayerNeiron();
            //    List<Neuron> ln = new List<Neuron>();
            //    for (int k = 0; k < staff.Layers[i].Count; k++)
            //    {
            //        Console.WriteLine(staff.Layers[i][k]);
            //        var neuron = JsonConvert.DeserializeObject<Neuron>(staff.Layers[i][k].ToString());
            //        ln.Add(neuron);
            //    }
            //    layer.Neurons = ln;
            //    lln.Add(layer);
            //}
            //Console.WriteLine("sd");
            //nn.Layers = lln;
            //string jsonAll = System.IO.File.ReadAllText($"{filename}.json");
            //string jsonLayers = "{" + jsonAll.Substring(jsonAll.IndexOf("\"Layers"));
            //var layers = JsonConvert.DeserializeObject<LayerNeiron>(jsonLayers);
            //nn= JsonConvert.DeserializeObject<NeuralNetwork>(jsonAll);
            //nn.Layers = layers;
            //return JsonConvert.DeserializeObject<NeuralNetwork>(jsonAll);
            return nn;
        }
    }
}
