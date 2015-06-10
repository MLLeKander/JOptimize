package nl.rug.joptimize.opt;

public interface OptParam<ParamType extends OptParam<ParamType>> {
    public ParamType add(ParamType o);

    public ParamType add_s(ParamType o);

    public ParamType sub(ParamType o);

    public ParamType sub_s(ParamType o);

    public ParamType zero();

    public ParamType zero_s();

    public ParamType dotprod(ParamType o);

    public ParamType dotprod_s(ParamType o);

    public ParamType multiply(double o);

    public ParamType multiply_s(double o);

    public ParamType copy();

    public double squaredNorm();
}
