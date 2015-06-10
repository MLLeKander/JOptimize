package nl.rug.joptimize.opt;


public class TimeOptObserver<ParamType extends OptParam<ParamType>> implements OptObserver<ParamType> {
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

    @Override
    public void notifyExample(ParamType params) {
        // NOOP
    }

}
