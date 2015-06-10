package nl.rug.joptimize.opt;

public interface Optimizer<ParamType extends OptParam<ParamType>> {
    public ParamType optimize(SeperableCostFunction<ParamType> ds, ParamType init);
}
