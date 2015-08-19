package nl.rug.joptimize.opt.optimizers;

import java.util.Collection;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class MultiBGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    protected double[] learningRates;
    
    public MultiBGD(Collection<Double> learningRates, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRates = new double[learningRates.size()];
        int i = 0;
        for (Double d : learningRates) {
            this.learningRates[i++] = d;
        }
    }

    public MultiBGD(double[] learningRates, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRates = learningRates;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType grad = cf.deriv(params);
        ParamType minParams = null;
        double minErr = Double.MAX_VALUE;
        double minRate = 0;
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
        System.out.println("\nminRate: "+minRate);
        return minParams;
    }
}