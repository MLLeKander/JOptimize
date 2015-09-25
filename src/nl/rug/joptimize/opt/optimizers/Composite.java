package nl.rug.joptimize.opt.optimizers;

import java.util.Collection;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class Composite<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> {
    private Collection<? extends Optimizer<ParamType>> bases;

    public Composite(Collection<? extends Optimizer<ParamType>> bases, double epsilon, int tMax) {
        super(epsilon, tMax);
        this.bases = bases;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType minParams = null;
        double minErr = Double.MAX_VALUE;
        Optimizer<?> opt = null;
        for (Optimizer<ParamType> base : bases) {
            ParamType tmpParams = base.optimizationStep(cf, params);
            double tmpErr = cf.error(tmpParams);
            if (tmpErr <= minErr) {
                minParams = tmpParams;
                minErr = tmpErr;
                opt = base;
            }
        }
        System.out.println("Best optimizer="+opt+", err="+minErr);
        return minParams;
    }

    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        for (Optimizer<ParamType> base : bases) {
            base.init(cf, initParams);
        }
    }

}
