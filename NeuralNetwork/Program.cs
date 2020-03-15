using System;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            new NeuralNet(new NeuralNetConfig
            {
                Input = 2,
                Target = 1,
                Hidden = 10,
                Rate = 0.1f,
                Momentum = 0.5f,
                Epocs = 100
            }).Train();
        }
    }
}
