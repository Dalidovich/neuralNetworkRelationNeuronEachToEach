namespace neuralNetworkRelationNeuronEachToEach
{
    public class Neuron
    {
        public int OutputId;
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputCount,int outputId, NeuronType type = NeuronType.Normal)
        {
            OutputId = outputId;
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValue(inputCount);
        }
        private void InitWeightsRandomValue(int inputCount)
        {
            //Console.WriteLine(inputCount);
            for (int i = 0; i < inputCount; i++)
            {
                var rnd = new Random();
                Thread.Sleep(100);
                var a = rnd.NextDouble() - 0.5;
                if (NeuronType == NeuronType.Input || NeuronType == NeuronType.bies)
                {
                    Weights.Add(1);
                }
                else
                {
                    //Console.WriteLine($"create weight={a}. i={i},type={NeuronType},id={OutputId},otput={Output}");
                    //Console.WriteLine(a);
                    Weights.Add(a);
                }
                Inputs.Add(1);
            }
        }
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                //MessageBox.Show(NeuronType.ToString()+"\n"+inputs[i].ToString()+"\n"+ Inputs.Count+"\n"+inputs.Count);
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input || NeuronType==NeuronType.bies)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input || NeuronType==NeuronType.bies)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeigth = weight - input * Delta * learningRate;
                Weights[i] = newWeigth;
            }
        }
        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
