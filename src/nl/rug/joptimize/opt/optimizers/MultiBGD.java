package nl.rug.joptimize.opt.optimizers;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class MultiBGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    protected double[] learningRates;

    public MultiBGD(double[] learningRates, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRates = learningRates;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType grad = cf.deriv(params);
        ParamType minParams = null;
        double minErr = Double.MAX_VALUE;
        for (double learningRate : learningRates) {
            ParamType tmpParams = grad.multiply_s(-learningRate).add_s(params);
            double tmpErr = cf.error(tmpParams);
            if (tmpErr <= minErr) {
                minParams = tmpParams;
                minErr = tmpErr;
            }
        }
        return minParams;
    }
}
