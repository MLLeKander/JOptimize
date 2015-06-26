package nl.rug.joptimize.opt.optimizers;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class BGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    protected double learningRate;

    public BGD(double learningRate, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRate = learningRate;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        //ParamType grad = cf.deriv(params);
        //return grad.multiply_s(-learningRate).add_s(params);
        return cf.deriv(params).multiply_s(-learningRate).add_s(params);
    }
    
    @Override
    public String toString() {
        return String.format("%s (nu=%.2f,ep=%.2f,tMax=%d)", this.getClass().getSimpleName(),learningRate,epsilon,tMax);
    }
}
