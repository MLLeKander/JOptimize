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
    
    public static double pDbl(Map<String, String> params, String key) {
        if (!params.containsKey("--"+key)) {
            throw new IllegalArgumentException("Required argument: "+key);
        }
        return Double.parseDouble(params.get("--"+key));
    }
    
    public static double pDbl(Map<String, String> params, double deflt, String key) {
        if (!params.containsKey("--"+key)) {
            return deflt;
        }
        return Double.parseDouble(params.get("--"+key));
    }
    
    public static int pInt(Map<String, String> params, String key) {
        if (!params.containsKey("--"+key)) {
            throw new IllegalArgumentException("Required argument: "+key);
        }
        return Integer.parseInt(params.get("--"+key));
    }
    
    public static int pInt(Map<String, String> params, int deflt, String key) {
        if (!params.containsKey("--"+key)) {
            return deflt;
        }
        return Integer.parseInt(params.get("--"+key));
    }
}
