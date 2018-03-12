package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class SGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    protected double learningRate;
    protected double effectiveLearningRate;
    protected Random rand;
    protected int t = 0;

    public SGD(long seed, double learningRate, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRate = learningRate;
        this.rand = new Random(seed);
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        return sgdEpoch(cf, params, learningRate);
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType params) {
        t = 0;
    }
    
    protected ParamType sgdEpoch(SeparableCostFunction<ParamType> cf, ParamType params, double learningRate) {
        ParamType out = params.copy();
        int size = cf.size();
        effectiveLearningRate = learningRate / (1+ t/tMax);
        
        for (int i = 0; i < size; i++) {
            out.sub_s(cf.deriv(out, rand.nextInt(size)).multiply_s(effectiveLearningRate));
        }
        t++;
        return out;
    }
}
