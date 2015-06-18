package nl.rug.joptimize.opt;

public interface SeparableCostFunction<ParamType extends OptParam<ParamType>> extends
        CostFunction<ParamType> {
    public double error(ParamType params, int exampleNdx);

    public ParamType deriv(ParamType params, int exampleNdx);

    public ParamType deriv(ParamType params, int exampleNdx, ParamType out);

    public ParamType hesseDiag(ParamType params, int exampleNdx);

    public ParamType hesseDiag(ParamType params, int exampleNdx, ParamType out);

    public int size();
}
