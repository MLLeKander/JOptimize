package nl.rug.joptimize.opt.optimizers;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class ControlledBGD<ParamType extends OptParam<ParamType>> extends
        AbstractOptimizer<ParamType> {
    private double initLearningRate;
    private double learningRate;
    private double loss;
    private double gain;
    protected double prevErr;

	public ControlledBGD(double initialLearningRate, double loss, double gain, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.initLearningRate = initialLearningRate;
        this.loss = loss;
        this.gain = gain;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType params) {
    	super.init(cf, params);
    	this.learningRate = this.initLearningRate;
    	this.prevErr = Double.MAX_VALUE;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType out = cf.deriv(params).multiply_s(-learningRate).add_s(params);
        double err = cf.error(out);
        if (err > prevErr) {
        	learningRate /= loss;
        } else {
        	learningRate *= gain;
        }
        System.out.println(learningRate);
        return out;
    }
    
    @Override
    public String toString() {
        return String.format("%s (nu=%.2f,ep=%.2f,tMax=%d)", this.getClass().getSimpleName(),learningRate,epsilon,tMax);
    }
}
