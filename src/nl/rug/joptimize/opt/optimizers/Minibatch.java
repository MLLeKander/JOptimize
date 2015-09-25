package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class Minibatch<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double learningRate;
    private Random rand;
    private int batches;

    public Minibatch(long seed, double learningRate, int batches, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.learningRate = learningRate;
        this.rand = new Random(seed);
        this.batches = batches;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType out = params.copy(), tmp = params.zero();
        int size = cf.size();
        int batchRem = size%batches, batchSize = size/batches + 1;
        
        for (int i = 0; i < batches; i++) {
            if (i == batchRem) {
                batchSize--;
            }
            tmp.zero_s();
            for (int j = 0; j < batchSize; j++) {
                cf.deriv(out, rand.nextInt(size), tmp);
            }
            out.sub_s(tmp.multiply_s(learningRate));
        }
        return out;
    }
    
    @Override
    public String toString() {
        return "Minibatch"+batches;
    }
}
