package nl.rug.joptimize.opt;

public interface CostFunction<ParamType extends OptParam<ParamType>> {

    public double error(ParamType params);

    public ParamType deriv(ParamType params);

    public ParamType hesseDiag(ParamType params);

}