package nl.rug.joptimize.opt.optimizers;

import java.util.Map;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class BGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double learningRate;
    private double epsilon;
    private int tMax;
    
    public BGD(Map<String, String> p) {
        this(pDbl(p,"rate"), pDbl(p,"epsilon"), pInt(p,"tmax"));
    }

    public BGD(double learningRate, double epsilon, int tMax) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    // TODO This doesn't need to be Separable...
    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();

        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff >= epsilon; t++) {
            ParamType grad = cf.deriv(params);

            diff = grad.squaredNorm();
            params.sub_s(grad.multiply_s(learningRate));
            err = cf.error(params);
            this.notifyEpoch(params, err);
        }
        System.err.println(params);
        return params;
    }
    
    public String toString() {
        return String.format("%s (nu=%.2f,ep=%.2f,tMax=%d)", this.getClass().getSimpleName(),learningRate,epsilon,tMax);
    }
}
