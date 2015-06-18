package nl.rug.joptimize.opt;

public interface Optimizer<ParamType extends OptParam<ParamType>> {
    public ParamType optimize(SeparableCostFunction<ParamType> ds, ParamType init);
}
