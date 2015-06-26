package nl.rug.joptimize.opt.observers;

import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptParam;

public class PrintObserver<ParamType extends OptParam<ParamType>> implements
        OptObserver<ParamType> {

    @Override
    public void notifyEpoch(ParamType params, double error) {
        System.out.println(error+" " +params);
    }
}
