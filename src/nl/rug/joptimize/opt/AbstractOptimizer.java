package nl.rug.joptimize.opt;

import java.util.*;

public abstract class AbstractOptimizer<ParamType extends OptParam<ParamType>> implements
        Optimizer<ParamType> {
    protected List<OptObserver<ParamType>> obs = new ArrayList<OptObserver<ParamType>>();
    protected double epsilon;
    protected int tMax;

    public AbstractOptimizer() {  }
    
    public AbstractOptimizer(double epsilon, int tMax) {
        this.epsilon = epsilon;
        this.tMax = tMax;
    }

    public void addObs(OptObserver<ParamType> ob) {
        obs.add(ob);
    }

    protected void notifyEpoch(ParamType params, double error) {
        for (OptObserver<ParamType> ob : obs) {
            ob.notifyEpoch(params, error);
        }
    }
    
    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        
    }
    
    @Override
    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        init(cf, initParams);
        ParamType params = initParams.copy();

        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff >= epsilon; t++) {
            ParamType newParams = optimizationStep(cf, params);
            
            diff = params.sub_s(newParams).squaredNorm();
            //System.out.print(diff+",");
            err = cf.error(newParams);
            this.notifyEpoch(newParams, err);
            
            params = newParams;
        }
        return params;
    }
}
