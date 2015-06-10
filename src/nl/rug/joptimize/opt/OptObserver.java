package nl.rug.joptimize.opt;

public interface OptObserver<ParamType extends OptParam<ParamType>> {
    // TODO ?
    public void notifyEpoch(ParamType params, double error);

    public void notifyExample(ParamType params);
}
