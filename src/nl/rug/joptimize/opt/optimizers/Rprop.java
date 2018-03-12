package nl.rug.joptimize.opt.optimizers;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;


public class Rprop<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> {
    private double initDelta;
    private double loss, gain;
    private double maxDelta, minDelta;
    private ParamType prevGrad = null, delta = null;

    public Rprop(double initDelta, double maxDelta, double minDelta, double loss, double gain, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.initDelta = initDelta;
        this.minDelta = minDelta;
        this.maxDelta = maxDelta;
        this.loss = loss;
        this.gain = gain;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        prevGrad = initParams.zero();
        delta = initParams.one().multiply_s(initDelta);
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {

        ParamType grad = cf.deriv(params);

        for (int i = 0; i < grad.length(); i++) {
            double gradTmp = grad.get(i);
            double agreement = gradTmp*prevGrad.get(i), newDelta;
            if (agreement > 0) {
                delta.set(i, newDelta = Math.min(delta.get(i)*gain, maxDelta));
            } else {
                delta.set(i, newDelta = Math.max(delta.get(i)*loss, minDelta));
            }
            double newW = params.get(i) - Math.signum(gradTmp) * newDelta;
            grad.set(i, newW);
            prevGrad.set(i, gradTmp);
        }
/*
        get current grad
        for each dimension:
          if signs agree:
            delta = min(delta*gain, maxDelta)
            w = w - sign(grad) * delta
            prevGrad = grad
        
          if signs disagree:
            delta = max(delta*loss, minDelta)
            prevGrad = 0
          
          if prevGrad was 0:
            w = w - sign(grad) * delta
            prevGrad = grad
 */
        /*
        for (int i = 0; i < grad.length(); i++) {
            double gradTmp = grad.get(i);
            double agreement = gradTmp*prevGrad.get(i);
            if (agreement > 0) {
                double newDelta = Math.min(delta.get(i)*gain, maxDelta);
                delta.set(i, newDelta);
                double newW = params.get(i) - Math.signum(gradTmp) * newDelta;
                grad.set(i, newW);
                prevGrad.set(i, gradTmp);
            } else if (agreement < 0) {
                delta.set(i, Math.max(delta.get(i)*loss, minDelta));
                prevGrad.set(i, 0);
            } else {
                double newW = params.get(i) - Math.signum(gradTmp) * delta.get(i);
                grad.set(i, newW);
                prevGrad.set(i, gradTmp);
            }
        }*/
        
        return grad;
    }
    
    @Override
    public String toString() {
        return delta.simplifiedToString();
    }
}
