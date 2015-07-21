package nl.rug.joptimize.opt.optimizers;

import java.util.Collection;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class Composite<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> {
    private Collection<Optimizer<ParamType>> bases;

    public Composite(Collection<Optimizer<ParamType>> bases) {
        this.bases = bases;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType minParams = null;
        double minErr = Double.MAX_VALUE;
        for (Optimizer<ParamType> base : bases) {
            ParamType tmpParams = base.optimizationStep(cf, params);
            double tmpErr = cf.error(tmpParams);
            if (tmpErr <= minErr) {
                minParams = tmpParams;
                minErr = tmpErr;
            }
        }
        return minParams;
    }

    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        for (Optimizer<ParamType> base : bases) {
            base.init(cf, initParams);
        }
    }

}
