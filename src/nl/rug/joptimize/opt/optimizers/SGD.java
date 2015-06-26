package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class SGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double learningRate;
    private Random rand;

    public SGD(long seed, double learningRate, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRate = learningRate;
        this.rand = new Random(seed);
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType out = params.copy();
        int size = cf.size();
        
        for (int i = 0; i < size; i++) {
            out.sub_s(cf.deriv(out, rand.nextInt(size)).multiply_s(learningRate));
        }
        return out;
    }
}
