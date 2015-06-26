package nl.rug.joptimize.opt.observers;

import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptParam;

public class CountObserver<ParamType extends OptParam<ParamType>> implements
        OptObserver<ParamType> {
    public int epochCount = 0;

    @Override
    public void notifyEpoch(ParamType params, double error) {
        epochCount++;
    }

    public int getEpochCount() {
        return epochCount;
    }
}
