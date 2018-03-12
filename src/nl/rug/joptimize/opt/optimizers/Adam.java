package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class Adam<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    protected Random rand;
    protected ParamType m, v, update;
    protected int batchSize, t;
    protected double alpha, beta1, beta2, beta1Pow, beta2Pow;

    private final static double eps = 1e-8;

    public Adam(long seed, int batchSize, double alpha, double beta1, double beta2, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.rand = new Random(seed);
        this.batchSize = batchSize;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.alpha = alpha;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        m = initParams.zero();
        v = initParams.zero();
        t = 0;
        beta1Pow = beta1;
        beta2Pow = beta2;
    }
    
    protected void adamUpdate(ParamType params, ParamType grad) {
        t++;
        m.multiply_s(beta1).add_s(grad.multiply(1-beta1));
        v.multiply_s(beta2).add_s(grad.dotprod_s(grad).multiply_s(1-beta2));
        //ParamType mHat = m.multiply(1/(1-beta1Pow));
        //ParamType vHat = v.multiply(1/(1-beta2Pow));
        ParamType mHat = m.multiply(1/(1-Math.pow(beta1, t)));
        ParamType vHat = v.multiply(1/(1-Math.pow(beta2, t)));
        update = mHat.multiply_s(alpha).dotprod_s(vHat.sqrt_s().add_s(eps).inv_s()); 
        params.sub_s(update);
        beta1Pow *= beta1;
        beta2Pow *= beta2;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy(), grad = params.zero();
        
        int size = cf.size();
        if (size <= batchSize) {
            grad = cf.deriv(outParams);
            adamUpdate(outParams, grad);
        } else {
            for (int i = 0; i < size; i++) {
                cf.deriv(outParams, rand.nextInt(size), grad);
                //System.out.println("outer loop");
                if ((i+1) % batchSize == 0) {
                    adamUpdate(outParams, grad);
                    grad.zero_s();
                }
            }
            if (size % batchSize != 0) {
                adamUpdate(outParams, grad);
            }
        }
        
        
        return outParams;
    }
    
    @Override
    public String toString() {
        return update.simplifiedToString();
    }
}
