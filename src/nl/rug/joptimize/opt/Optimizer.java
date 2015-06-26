package nl.rug.joptimize.opt;

public interface Optimizer<ParamType extends OptParam<ParamType>> {
    public ParamType optimize(SeparableCostFunction<ParamType> cf, ParamType initParams);
    public void init(SeparableCostFunction<ParamType> cf, ParamType initParams);
    public ParamType optimizationStep(SeparableCostFunction<ParamType> cf, ParamType params);
}
