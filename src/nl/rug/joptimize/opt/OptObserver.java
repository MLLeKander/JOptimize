package nl.rug.joptimize.opt;

public interface OptObserver<ParamType extends OptParam<ParamType>> {
    public void notifyEpoch(ParamType params, double error);
}
