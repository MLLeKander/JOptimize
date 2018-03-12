package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class Adadelta<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    protected Random rand;
    protected ParamType newDelta;
    protected ParamType g, delta;
    protected int batchSize;
    protected double rho;

    private final static double eps = 1e-8;

    public Adadelta(long seed, int batchSize, double rho, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.rand = new Random(seed);
        this.batchSize = batchSize;
        this.rho = rho;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        newDelta = initParams;
        g = initParams.zero();
        delta = initParams.zero();
    }
    
    protected void adadeltaUpdate(ParamType params, ParamType grad) {
        g.multiply_s(rho).add_s(grad.dotprod(grad).multiply_s(1-rho));
        ParamType rmsGrad = g.add(eps).sqrt_s();
        ParamType rmsDelta = delta.add(eps).sqrt_s();
        newDelta = rmsDelta.dotprod_s(grad).dotprod_s(rmsGrad.inv_s());
        params.sub_s(newDelta);
        delta.multiply_s(rho).add_s(newDelta.dotprod(newDelta).multiply_s(1-rho));
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy(), grad = params.zero();
        
        int size = cf.size();
        if (size <= batchSize) {
            grad = cf.deriv(outParams);
            adadeltaUpdate(outParams, grad);
        } else {
            for (int i = 0; i < size; i++) {
                cf.deriv(outParams, rand.nextInt(size), grad);
                if ((i+1) % batchSize == 0) {
                    adadeltaUpdate(outParams, grad);
                    grad.zero_s();
                }
            }
            if (size % batchSize != 0) {
                adadeltaUpdate(outParams, grad);
            }
        }
        
        
        return outParams;
    }
    
    @Override
    public String toString() {
        return newDelta.simplifiedToString();
    }
}
