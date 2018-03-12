package nl.rug.joptimize.opt;

import java.util.*;

public abstract class AbstractOptimizer<ParamType extends OptParam<ParamType>> implements
        Optimizer<ParamType> {
    protected List<OptObserver<ParamType>> obs = new ArrayList<OptObserver<ParamType>>();
    protected double epsilon;
    protected int tMax;
    protected long nsMax, startTime;

    public AbstractOptimizer() {  }
    
    public AbstractOptimizer(double epsilon, int tMax, long nsMax) {
        this.epsilon = epsilon;
        this.tMax = tMax;
        this.nsMax = nsMax;
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
    
    protected boolean elapsed() {
        return nsMax > 0 && nsMax < System.nanoTime() - startTime;
    }
    
    @Override
    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        init(cf, initParams);
        ParamType params = initParams.copy();

        startTime = System.nanoTime();
        double err = cf.error(params), diff = Double.MAX_VALUE;
        for (int t = 0; t < tMax && diff >= epsilon && !elapsed(); t++) {
            ParamType newParams = optimizationStep(cf, params);
            
            diff = params.sub_s(newParams).squaredNorm();
            err = cf.error(newParams);
            this.notifyEpoch(newParams, err);
            
            params = newParams;
        }
        return params;
    }
}
