package nl.rug.joptimize.opt.optimizers;

import java.util.Map;
import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class vSGD<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    private double epsilon;
    private int tMax;
    private Random rand = new Random();
    
    public vSGD(Map<String, String> p) {
        this(pDbl(p,"epsilon"),pInt(p,"tmax"));
    }

    public vSGD(double epsilon, int tMax) {
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();
        int size = cf.size();
        int C = 10;
        double eps = 1e-10;
        ParamType ONE = initParams.one(), EPS = ONE.multiply(eps);
        
        ParamType gs = params.zero(), vs = params.zero();
        ParamType taus = params.one().multiply_s(size + eps);
        
        for (int i = 0; i < size; i++) {
            ParamType partialGrad = cf.deriv(params, i);
            gs.add_s(partialGrad);
            vs.add_s(partialGrad.dotprod_s(partialGrad));
        }
        
        gs.multiply_s(1./size);
        vs.multiply_s(C/(double)size);
        ParamType hs = cf.hesseDiag(params).multiply_s(C/(double)size).abs_s();
        //System.out.printf("init gs:%s\ninit vs:%s\ninit hs:%s\ntaus: %s\n", gs,vs,hs,taus);

        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff > epsilon; t++) {
            ParamType init = params.copy();
            for (int i = 0; i < size; i++) {
                int sampleNdx = rand.nextInt(size);
                ParamType grad = cf.deriv(params, sampleNdx);
                ParamType hesse = cf.hesseDiag(params, sampleNdx);
                
                ParamType tauInv = taus.inv();
                ParamType tauInvComp = taus.one().sub_s(tauInv);

                // gs = tauInvComp .* gs + tauInv .* gradTmp
                gs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad));
                // vs = tauInvComp .* vs + tauInv .* (gradTmp .^ 2)
                vs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad).dotprod_s(grad));
                // hs = tauInvComp .* hs + tauInv .* max(epsilon,hesse)
                // TODO: Max or abs?
                hs.dotprod_s(tauInvComp).add_s(tauInv.dotprod_s(hesse).abs_s());
                
                // TODO: .divide?
                
                ParamType learningRates = gs.dotprod(gs).dotprod_s(hs.dotprod(vs).add_s(EPS).inv_s());
                params.sub_s(grad.dotprod_s(learningRates));
                //System.out.printf("grad:%s\nhess:%s\ngs:%s\nvs:%s\nhs:%s\nrates:%s", grad,hesse,gs,vs,hs,learningRates);
                
                // taus = (1 - (gs.^2)./vs).*taus + 1
                taus.dotprod_s(vs.inv().dotprod_s(gs).dotprod_s(gs).multiply_s(-1).add_s(ONE)).add_s(ONE);

                this.notifyExample(grad);
            }
            err = cf.error(params);
            this.notifyEpoch(params, err);
            diff = init.sub_s(params).squaredNorm();
        }
        return params;
    }
}
