using System;

namespace NeuralNetwork
{
    class NeuralNet
    {
        #region fields
        private static readonly Random random = new Random();
        private static int iteration = 0;
        #endregion

        #region instance vars
        //private float[] input, output;
        private double[,] wHidden, wTarget;
        private double[,] dwHidden, dwTarget;
        private double[] oHidden;
        private double[] oTarget;
        private double[] eTarget;
        private double mse;
        private double hBias, tBias = 1;
        private NeuralNetConfig config;
        #endregion

        public NeuralNet(NeuralNetConfig config)
        {
            //input = new float[config.Input];
            //output = new float[config.Output];
            wHidden = new double[config.Hidden /*+ 1*/, config.Input];
            wTarget = new double[config.Target, config.Hidden/*+1*/];
            oHidden = new double[config.Hidden/*+1*/];
            oTarget = new double[config.Target];
            dwTarget = new double[config.Target, config.Hidden/*+1*/];
            dwHidden = new double[config.Hidden/*+1*/, config.Input/*+1*/];
            eTarget = new double[config.Target];
            this.config = config;
        }

        public void Train()
        {
            InitializeWeights();
            double msed = 0;
            for (int k = 0; k < config.Epocs; k++)
            {
                msed = 0;
                for (int i = 0; i < 5000; i++)
                {
                    Backpropagete(new double[] { Sigmoid(i), Sigmoid(i + 1) },
                        new double[] { Sigmoid(2 * i + 3 * (i + 1)) });
                    msed += mse;
                    //Console.Out.WriteLine(msed);
                }
                msed *= 0.5;
                //Console.Out.WriteLine(msed);
            }
            Console.Out.WriteLine(msed);
            Compute(new double[] { Sigmoid(2), Sigmoid(2) }, new double[] { Sigmoid(10) });
            double actual = oTarget[0];
            double expected = Sigmoid(10);
            Console.WriteLine("actual: " + actual + " expected: " + expected);
        }

        void Backpropagete(double[] input, double[] target)
        {
            //Array.Resize(ref input, input.Length+1);
            //input[input.Length] = 1;    //bias
            PropagteHidden(input);
            PropagateTarget(target);
            AdjustTarget(target);
            AdjustHidden(input);
            iteration++;
        }

        void Compute(double[] input, double[] target)
        {
            PropagteHidden(input);
            PropagateTarget(target);
        }

        private void PropagteHidden(double[] input)
        {
            for (int i = 0; i < config.Hidden; i++)
            {
                oHidden[i] = Sigmoid(WeightedSum(i, input, wHidden));
            }
        }

        private void PropagateTarget(double[] target)
        {
            mse = 0;
            for (int i = 0; i < config.Target; i++)
            {
                oTarget[i] = Sigmoid(WeightedSum(i, oHidden, wTarget) + tBias);
                mse += SquaredError(target[i], oTarget[i]);
            }
        }

        private void AdjustTarget(double[] target)
        {
            for (int i = 0; i < config.Target; i++)
            {
                eTarget[i] = (target[i] - oTarget[i]) * oTarget[i] * (1 - oTarget[i]);
                for (int j = 0; j < config.Hidden; j++)
                {
                    double delta = DeltaWeight(eTarget[i], oHidden[j]);
                    dwTarget[i, j] = NewDeltaWeight(dwTarget[i, j], delta);
                    wTarget[i, j] += dwTarget[i, j];
                }
            }
        }

        private void AdjustHidden(double[] input)
        {
            for (int i = 0; i < config.Hidden; i++)
            {
                for (int j = 0; j < config.Input; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < config.Target; k++)
                        sum += eTarget[k] * wTarget[k, i];
                    double delta = DeltaWeight(oHidden[i] * (1 - oHidden[i]) * sum, input[j]);
                    dwHidden[i, j] = NewDeltaWeight(dwHidden[i, j], delta);
                    wHidden[i, j] += dwHidden[i, j];
                }
            }
        }

        private void InitializeWeights()
        {
            IntializeWeights(wHidden/*, 0, 1*/);
            IntializeWeights(wTarget/*, 0, 1*/);
        }

        private double NewDeltaWeight(double previous, double deltaWeight)
        {
            return iteration == 0 ? deltaWeight : config.Momentum * previous + deltaWeight;
        }

        private double DeltaWeight(double del, double input)
        {
            return config.Rate * del * input;
        }

        private double WeightedSum(int nodeIndex, double[] input, double[,] weights)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += weights[nodeIndex, i] * input[i];
            }
            return sum;
        }

        private static double SquaredError(double target, double actual)
        {
            return Math.Pow(target - actual, 2);
        }

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private static void IntializeWeights(double[,] weights)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
                for (int j = 0; j < weights.GetLength(1); j++)
                    weights[i, j] = random.NextDouble();

            //bias
            //for (int i = 0; i < weights.GetLength(1); i++)
              //  weights[weights.GetLength(0) - 1, i] = 1;
        }
    }
}
