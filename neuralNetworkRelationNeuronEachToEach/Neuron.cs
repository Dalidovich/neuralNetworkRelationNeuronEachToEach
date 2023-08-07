namespace neuralNetworkRelationNeuronEachToEach
{
    public class Neuron
    {
        public int id { get; set; }
        public int OutputId;
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int id,int inputCount,int outputId, NeuronType type = NeuronType.Normal)
        {
            OutputId = outputId;
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            this.id = id;
            InitWeightsRandomValue(inputCount);
            if (type == NeuronType.bies)
            {
                Output = 1;
            }
        }

        public Task initTask;

        private void InitWeightsRandomValue(int inputCount)
        {
            initTask=new Task(() =>
            {
                for (int i = 0; i < inputCount; i++)
                {
                    Thread.Sleep(100);
                    if (NeuronType == NeuronType.Input || NeuronType == NeuronType.bies)
                    {
                        Weights.Add(1);
                    }
                    else
                    {
                        if (i != inputCount - 1)
                        {
                            var a = randomForTest.v();
                            Weights.Add(a);
                        }
                        else
                        {
                            Weights.Add(1.0);
                        }
                    }
                    Inputs.Add(1);
                }
            });
            
        }

        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if (NeuronType == NeuronType.bies)
                return 1;
            if (NeuronType != NeuronType.Input && NeuronType!=NeuronType.bies)
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

        public void Learn(double error, double learningRate,double neuronNum=0)
        {
            if (NeuronType == NeuronType.Input || NeuronType == NeuronType.bies)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];
                    
                var newWeigth = weight + input * Delta * learningRate;
                Weights[i] = newWeigth;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
