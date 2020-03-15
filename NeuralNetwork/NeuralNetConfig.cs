namespace NeuralNetwork
{
    class NeuralNetConfig
    {
        public int Input { get; set; }
        public int Target { get; set; }
        public int Hidden { get; set; }
        public float Rate { get; set; }
        public float Momentum { get; set; }

        public int Epocs { get; set; }
    }
}
