package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class VSGD<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    private Random rand;
    private int batchSize;
    private ParamType gs, vs, taus, hs, ONE, learningRates;

    private final static double eps = 1e-10;
    private final static int C = 10;

    public VSGD(long seed, int batchSize, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.rand = new Random(seed);
        this.batchSize = batchSize;
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        int size = cf.size();
        gs = initParams.zero();
        vs = initParams.zero();
        
        for (int i = 0; i < size; i++) {
            ParamType partialGrad = cf.deriv(initParams, i);
            gs.add_s(partialGrad);
            vs.add_s(partialGrad.dotprod_s(partialGrad));
        }
        
        gs.multiply_s(1./size);
        vs.multiply_s(C/(double)size);
        taus = initParams.one().multiply_s(size + eps);
        hs = cf.hesseDiag(initParams).multiply_s(C/(double)size).abs_s();
        ONE = initParams.one();
    }

    public void vsgdUpdate(ParamType params, ParamType grad, ParamType hesse) {
        ParamType tauInv = taus.inv();
        ParamType tauInvComp = taus.one().sub_s(tauInv);

        // gs = tauInvComp .* gs + tauInv .* gradTmp
        gs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad));
        // vs = tauInvComp .* vs + tauInv .* (gradTmp .^ 2)
        vs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad).dotprod_s(grad).lbound_s(eps));
        // hs = tauInvComp .* hs + tauInv .* max(epsilon,hesse)
        hs.dotprod_s(tauInvComp).add_s(tauInv.dotprod_s(hesse.lbound_s(eps)));
        
        learningRates = gs.dotprod(gs).dotprod_s(hs.dotprod(vs).lbound_s(eps).inv_s());
        params.sub_s(grad.dotprod(learningRates));
        
        // taus = (1 - (gs.^2)./vs).*taus + 1
        taus.dotprod_s(vs.inv().dotprod_s(gs).dotprod_s(gs).multiply_s(-1).add_s(ONE)).add_s(ONE);
    }
    
    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy(), grad, hesse;
        
        int size = cf.size();
        if (size <= batchSize) {
            grad = cf.deriv(outParams);
            hesse = cf.hesseDiag(outParams);
            vsgdUpdate(outParams, grad, hesse);
        } else {
            grad = params.zero();
            hesse = params.zero();
            for (int i = 0; i < size; i++) {
                int ndx = rand.nextInt(size);
                cf.deriv(outParams, ndx, grad);
                cf.hesseDiag(outParams, ndx, hesse);
                if ((i+1) % batchSize == 0) {
                    vsgdUpdate(outParams, grad, hesse);
                    grad.zero_s();
                    hesse.zero_s();
                }
            }
            if (size % batchSize != 0) {
                vsgdUpdate(outParams, grad, hesse);
            }
        }

        return outParams;
    }
    
    @Override
    public String toString() {
        return learningRates.simplifiedToString();
    }
}
