package nl.rug.joptimize.opt.optimizers;

import java.util.Collection;

import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class SlowStartSGD<ParamType extends OptParam<ParamType>> extends SGD<ParamType> {
    protected double[] initLearningRates;
    
    public SlowStartSGD(Collection<Double> learningRates, long seed, double epsilon, int tMax) {
        super(seed, -1, epsilon, tMax);
        int i = 0;
        for (Double d : learningRates) {
            this.initLearningRates[i++] = d;
        }
    }
    
    public SlowStartSGD(double[] learningRates, long seed, double epsilon, int tMax) {
        super(seed, -1, epsilon, tMax);
        this.initLearningRates = learningRates;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType params) {
        double minErr = Double.MAX_VALUE;
        double minLearningRate = -1;
        for (double tmpLearningRate : initLearningRates) {
            ParamType tmpParams = sgdEpoch(cf, params, tmpLearningRate);
            t = 0;
            double tmpErr = cf.error(tmpParams); 
            if (tmpErr < minErr) {
                minErr = tmpErr;
                minLearningRate = tmpLearningRate;
            }
        }
        this.learningRate = minLearningRate;
    }
}
