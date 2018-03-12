package nl.rug.joptimize.opt.optimizers;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class VSGD<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> { 
    private Random rand;
    private ParamType gs, vs, taus, hs, ONE, learningRates;

    private final static double eps = 1e-10;
    private final static int C = 10;

    public VSGD(long seed, double epsilon, int tMax) {
        super(epsilon, tMax);
        rand = new Random(seed);
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        super.init(cf, initParams);
        int size = cf.size();
        gs = initParams.zero();
        vs = initParams.zero();
        
        //System.out.println("init:\n"+initParams);
        for (int i = 0; i < size; i++) {
            ParamType partialGrad = cf.deriv(initParams, i);
            //ParamType partialHesse = cf.hesseDiag(initParams, i);
            //System.out.println(i+1+"\n"+partialHesse);
            gs.add_s(partialGrad);
            vs.add_s(partialGrad.dotprod_s(partialGrad));
        }
        
        gs.multiply_s(1./size);
        vs.multiply_s(C/(double)size);
        taus = initParams.one().multiply_s(size + eps);
        hs = cf.hesseDiag(initParams).multiply_s(C/(double)size).abs_s();
        //System.out.println("g\n"+gs);
        //System.out.println("h\n"+hs);
        ONE = initParams.one();
//        System.out.printf("init gs:%s\ninit vs:%s\ninit hs:%s\ntaus: %s\nONE: %s\n", gs,vs,hs,taus,ONE);
    }
/*
    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        ParamType params = initParams.copy();
        int size = cf.size();
        int C = 10;
        ParamType ONE = initParams.one();
        
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
//      System.out.printf("init gs:%s\ninit vs:%s\ninit hs:%s\ntaus: %s\n", gs,vs,hs,taus);

        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff > epsilon; t++) {
//            System.out.println(" --- START EPOCH ---");
            ParamType init = params.copy();
            for (int i = 0; i < size; i++) {
//                System.out.println(" --- START EXAMPLE ---");
                int sampleNdx = rand.nextInt(size);
                ParamType grad = cf.deriv(params, sampleNdx);
                ParamType hesse = cf.hesseDiag(params, sampleNdx);

//                System.out.printf("grad:%s\nhesse:%s\n",grad,hesse);
                
                ParamType tauInv = taus.inv();
                ParamType tauInvComp = taus.one().sub_s(tauInv);

                // gs = tauInvComp .* gs + tauInv .* gradTmp
                gs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad));
                // vs = tauInvComp .* vs + tauInv .* (gradTmp .^ 2)
                vs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad).dotprod_s(grad));
                // hs = tauInvComp .* hs + tauInv .* max(epsilon,hesse)
                hs.dotprod_s(tauInvComp).add_s(tauInv.dotprod_s(hesse.lbound_s(eps)));
                
                // TODO: .divide?
                
                ParamType learningRates = gs.dotprod(gs).dotprod_s(hs.dotprod(vs).lbound_s(eps).inv_s());
                params.sub_s(grad.dotprod(learningRates));
                
                // taus = (1 - (gs.^2)./vs).*taus + 1
//                System.out.println("subA: "+subA);
                taus.dotprod_s(vs.inv().dotprod_s(gs).dotprod_s(gs).multiply_s(-1).add_s(ONE)).add_s(ONE);

//                System.out.printf("taus:%s\ngs:%s\nvs:%s\nhs:%s\nrates:%s\nsubA:%s\nparams:%s\n",taus,gs,vs,hs,learningRates,subA,params);
//                System.out.println(" --- END EXAMPLE ---");
            }
            err = cf.error(params);
            this.notifyEpoch(params, err);
            diff = init.sub_s(params).squaredNorm();
//            System.out.println(" --- END EPOCH ---");
        }
        return params;
    }
*/
    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = params.copy();
        int size = cf.size();
        
        for (int i = 0; i < size; i++) {
//          System.out.println(" --- START EXAMPLE ---");
          int sampleNdx = rand.nextInt(size);
          ParamType grad = cf.deriv(outParams, sampleNdx);
          ParamType hesse = cf.hesseDiag(outParams, sampleNdx);

//          System.out.printf("grad: %s\nhesse:%s\n",grad,hesse);
          
          ParamType tauInv = taus.inv();
          ParamType tauInvComp = taus.one().sub_s(tauInv);

          // gs = tauInvComp .* gs + tauInv .* gradTmp
          gs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad));
          // vs = tauInvComp .* vs + tauInv .* (gradTmp .^ 2)
          vs.dotprod_s(tauInvComp).add_s(tauInv.dotprod(grad).dotprod_s(grad).lbound_s(eps));
          // hs = tauInvComp .* hs + tauInv .* max(epsilon,hesse)
          hs.dotprod_s(tauInvComp).add_s(tauInv.dotprod_s(hesse.lbound_s(eps)));
          
          // TODO: .divide?
          
          learningRates = gs.dotprod(gs).dotprod_s(hs.dotprod(vs).lbound_s(eps).inv_s());
          outParams.sub_s(grad.dotprod(learningRates));
          
          // taus = (1 - (gs.^2)./vs).*taus + 1
//          ParamType tmp1 = vs.inv().dotprod_s(gs).dotprod_s(gs);
//          ParamType tmp2 = tmp1.multiply_s(-1).add_s(ONE);
//          System.out.println("taus1:"+taus);
//          System.out.println("vs:   "+vs);
//          System.out.println("vs':  "+vs.inv());
//          System.out.println("tmp1: "+tmp1);
//          System.out.println("tmp2: "+tmp2);
          taus.dotprod_s(vs.inv().dotprod_s(gs).dotprod_s(gs).multiply_s(-1).add_s(ONE)).add_s(ONE);

//          System.out.printf("taus: %s\ngs:   %s\nvs:   %s\nhs:   %s\nrates:%s\nparams:%s\n",taus,gs,vs,hs,learningRates,params);
//          System.out.println(" --- END EXAMPLE ---");
      }

        return outParams;
    }
    
    @Override
    public String toString() {
        return learningRates.simplifiedToString();
    }
}
