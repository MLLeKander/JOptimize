package nl.rug.joptimize.opt;

import java.util.*;

public abstract class AbstractOptimizer<ParamType extends OptParam<ParamType>> implements
        Optimizer<ParamType> {
    protected List<OptObserver<ParamType>> obs = new ArrayList<OptObserver<ParamType>>();

    public void addObs(OptObserver<ParamType> ob) {
        obs.add(ob);
    }

    protected void notifyExample(ParamType params) {
        for (OptObserver<ParamType> ob : obs) {
            ob.notifyExample(params);
        }
    }

    protected void notifyEpoch(ParamType params, double error) {
        for (OptObserver<ParamType> ob : obs) {
            ob.notifyEpoch(params, error);
        }
    }
}
