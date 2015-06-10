package nl.rug.joptimize.opt;

import java.util.Random;

public class SGDOptimizer<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double learningRate;
    private double epsilon;
    private int tMax;
    private Random rand = new Random();

    public SGDOptimizer(double learningRate, double epsilon, int tMax) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    public ParamType optimize(SeperableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();
        int size = cf.size();

        double err = cf.error(params);
        for (int t = 0; t < tMax && err < epsilon; t++) {
            for (int i = 0; i < size; i++) {
                ParamType partialGrad = cf.deriv(params, rand.nextInt(size));
                params.add_s(partialGrad.multiply_s(learningRate));

                this.notifyExample(partialGrad);
            }
            err = cf.error(params);
            this.notifyEpoch(params, err);
        }
        return params;
    }
}
