package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class ControlledSGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double initLearningRate;
    private double learningRate;
    private double loss;
    private double gain;
    protected double prevErr;
    private Random rand;

	public ControlledSGD(long seed, double initialLearningRate, double loss, double gain, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.initLearningRate = initialLearningRate;
        this.loss = loss;
        this.gain = gain;
        this.rand = new Random(seed);
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType params) {
    	super.init(cf, params);
    	this.learningRate = this.initLearningRate;
    	this.prevErr = Double.MAX_VALUE;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType out = params.copy();
        int size = cf.size();
        
        for (int i = 0; i < size; i++) {
            out.sub_s(cf.deriv(out, rand.nextInt(size)).multiply_s(learningRate));
        }
        double err = cf.error(out);
        if (err > prevErr) {
        	learningRate /= loss;
        } else {
        	learningRate *= gain;
        }
        return out;
    }
    
    @Override
    public String toString() {
        return String.format("%s (nu=%.2f,ep=%.2f,tMax=%d)", this.getClass().getSimpleName(),learningRate,epsilon,tMax);
    }
}
