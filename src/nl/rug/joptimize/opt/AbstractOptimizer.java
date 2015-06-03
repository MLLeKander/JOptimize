package nl.rug.joptimize.opt;

import java.util.*;

public abstract class AbstractOptimizer implements Optimizer {
    protected List<OptObserver> obs = new ArrayList<OptObserver>();

    public void addObs(OptObserver ob) {
        obs.add(ob);
    }

    protected void notifyExample(OptParam params) {
        for (OptObserver ob : obs) {
            ob.notifyExample(params);
        }
    }

    protected void notifyEpoch(OptParam params) {
        for (OptObserver ob : obs) {
            ob.notifyEpoch(params);
        }
    }
}
