package nl.rug.joptimize.opt.optimizers;

import java.util.ArrayDeque;

import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class WaypointAverage<ParamType extends OptParam<ParamType>> extends AbstractOptimizer<ParamType> {
    private int histSize;
    private double histInv;
    private ArrayDeque<ParamType> hist;
    private Optimizer<ParamType> base;
    private ParamType runningSum;

    public WaypointAverage(Optimizer<ParamType> base, int histSize, double epsilon, int tMax, long nsMax) {
        super(epsilon, tMax, nsMax);
        this.histSize = histSize;
        this.histInv = 1./histSize;
        this.base = base;
    }

    @Override
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params) {
        ParamType outParams = base.optimizationStep(cf, params);
        
        //TODO: .multiply(histInv) before adding to runningSum?
        runningSum.add_s(outParams);
        if (hist.size() >= histSize-1) {
            ParamType waypointAverage = runningSum.multiply(histInv);
            
            //TODO: Wasting a full error calculation here...
            //TODO: Learning rate control?
            if (cf.error(waypointAverage) < cf.error(outParams)) {
                runningSum.sub_s(outParams).add_s(waypointAverage);
                outParams = waypointAverage.copy();
            }
            runningSum.sub_s(hist.remove());
        }
        hist.add(outParams.copy());
        return outParams;
    }

    @Override
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams) {
        this.runningSum = initParams.zero();
        this.hist = new ArrayDeque<>(histSize);
        base.init(cf, initParams);
    }

}
