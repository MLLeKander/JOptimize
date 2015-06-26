package nl.rug.joptimize.opt.observers;

import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.OptParam;


public class TimeObserver<ParamType extends OptParam<ParamType>> implements OptObserver<ParamType> {
    public boolean isFirst = true;
    public long start, end;
    
    @Override
    public void notifyEpoch(ParamType params, double error) {
        this.end = System.nanoTime();
        if (this.isFirst) {
            this.start = this.end;
            this.isFirst = false;
        }
    }
}
