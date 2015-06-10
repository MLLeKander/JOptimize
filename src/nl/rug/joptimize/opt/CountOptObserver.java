package nl.rug.joptimize.opt;

public class CountOptObserver<ParamType extends OptParam<ParamType>> implements
        OptObserver<ParamType> {
    public int epochCount = 0, exampleCount = 0;

    @Override
    public void notifyEpoch(ParamType params, double error) {
        epochCount++;
    }

    @Override
    public void notifyExample(ParamType params) {
        exampleCount++;
    }

    public int getExampleCount() {
        return exampleCount;
    }

    public int getEpochCount() {
        return epochCount;
    }
}
