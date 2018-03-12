package nl.rug.joptimize.opt;

import java.util.Random;

public interface OptParam<ParamType extends OptParam<ParamType>> {
    public ParamType add(ParamType o);

    public ParamType add_s(ParamType o);

    public ParamType sub(ParamType o);

    public ParamType sub_s(ParamType o);

    public ParamType zero();

    public ParamType zero_s();

    public ParamType one();

    public ParamType one_s();

    public ParamType dotprod(ParamType o);

    public ParamType dotprod_s(ParamType o);

    public ParamType sqrt();

    public ParamType sqrt_s();

    public ParamType inv();

    public ParamType inv_s();

    public ParamType abs();

    public ParamType abs_s();

    public ParamType lbound(double o);

    public ParamType lbound_s(double o);

    public ParamType ubound(double o);

    public ParamType ubound_s(double o);

    public ParamType add(double o);

    public ParamType add_s(double o);

    public ParamType multiply(double o);

    public ParamType multiply_s(double o);

    public ParamType copy();

    public double squaredNorm();

    public ParamType random(Random r);
    
    public ParamType random_s(Random r);
    
    public String simplifiedToString();
}
