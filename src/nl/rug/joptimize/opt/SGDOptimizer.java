package nl.rug.joptimize.opt;

import java.util.Random;

public class SGDOptimizer extends AbstractOptimizer {
    private double learningRate;
    private double epsilon;
    private int tMax;
    private Random rand = new Random();

    public SGDOptimizer(double learningRate, double epsilon, int tMax) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    public OptParam optimize(SeperableCostFunction ds, OptParam initParams) {
        OptParam params = initParams;
        int size = ds.size();

        for (int t = 0; t < tMax && ds.error(params) < epsilon; t++) {
            OptParam partialGrad = ds.deriv(params, rand.nextInt(size));
            params.add_s(partialGrad.multiply_s(learningRate));

            this.notifyExample(partialGrad);
            if (t % size == 0) {
                this.notifyEpoch(params);
            }
        }
        return params;
    }
}
