package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class RMSprop<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    protected Random rand;
    protected ParamType v;
    protected int batchSize;
    protected double learningRate, rho;

    private final static double eps = 1e-8;

    public RMSprop(long seed, int batchSize, double learningRate, double rho, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.rand = new Random(seed);
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.rho = rho;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        v = initParams.zero();
    }
    
    protected void rmspropUpdate(ParamType params, ParamType grad) {
        v.multiply_s(rho).add_s(grad.dotprod(grad).multiply_s(1-rho));
        ParamType update = grad.multiply_s(learningRate).dotprod_s(v.sqrt().add_s(eps).inv_s());
        params.sub_s(update);
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy(), grad = params.zero();
        
        int size = cf.size();
        if (size <= batchSize) {
            grad = cf.deriv(outParams);
            rmspropUpdate(outParams, grad);
        } else {
            for (int i = 0; i < size; i++) {
                cf.deriv(outParams, rand.nextInt(size), grad);
                if ((i+1) % batchSize == 0) {
                    rmspropUpdate(outParams, grad);
                    grad.zero_s();
                }
            }
            if (size % batchSize != 0) {
                rmspropUpdate(outParams, grad);
            }
        }
        
        
        return outParams;
    }
    
    @Override
    public String toString() {
        return v.inv().multiply_s(learningRate).simplifiedToString();
    }
}
