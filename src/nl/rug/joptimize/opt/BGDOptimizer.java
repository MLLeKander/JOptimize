package nl.rug.joptimize.opt;

public class BGDOptimizer<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double learningRate;
    private double epsilon;
    private int tMax;

    public BGDOptimizer(double learningRate, double epsilon, int tMax) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    // TODO This doesn't need to be Seperable...
    public ParamType optimize(SeperableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();

        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff >= epsilon; t++) {
            ParamType grad = cf.deriv(params);

            diff = grad.squaredNorm();
            params.sub_s(grad.multiply_s(learningRate));
            err = cf.error(params);
            this.notifyEpoch(params, err);
        }
        return params;
    }
}
