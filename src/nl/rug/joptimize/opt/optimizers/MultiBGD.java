package nl.rug.joptimize.opt.optimizers;

import java.util.Collection;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class MultiBGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    protected double[] learningRates;
    public double minErr, minRate;
    
    public MultiBGD(Collection<Double> learningRates, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.learningRates = new double[learningRates.size()];
        int i = 0;
        for (Double d : learningRates) {
            this.learningRates[i++] = d;
        }
    }

    public MultiBGD(double[] learningRates, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.learningRates = learningRates;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType grad = cf.deriv(params);
        ParamType minParams = null;
        minErr = Double.MAX_VALUE;
        minRate = 0;
        for (double learningRate : learningRates) {
            ParamType tmpParams = grad.multiply(-learningRate).add_s(params);
            double tmpErr = cf.error(tmpParams);
            if (tmpErr <= minErr) {
                minParams = tmpParams;
                minErr = tmpErr;
                minRate = learningRate;
            }
            //System.out.printf("%.2f:%.4f ",learningRate,tmpErr);
        }
        return minParams;
    }
}
