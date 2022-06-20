namespace neuralNetworkRelationNeuronEachToEach
{
    public class LayerNeiron
    {
        public List<Neuron> Neurons { get; set; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type;
        public LayerNeiron(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            // TODO: проверить все входные нейроны на соответствие типу

            Neurons = neurons;
            Type = type;
        }

        public LayerNeiron()
        {
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
    }
}
