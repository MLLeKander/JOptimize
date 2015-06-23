package nl.rug.joptimize.opt;

public abstract class AbstractOptParam<ParamType extends OptParam<ParamType>> implements OptParam<ParamType> {
	public ParamType add(ParamType o) { return this.copy().add_s(o); }

    public ParamType sub(ParamType o) { return this.copy().sub_s(o); }

    public ParamType zero() { return this.copy().zero_s(); }

    public ParamType one() { return this.copy().one_s(); }

    public ParamType dotprod(ParamType o) { return this.copy().dotprod_s(o); }

    public ParamType inv() { return this.copy().inv_s(); }

    public ParamType abs() { return this.copy().abs_s(); }

    public ParamType multiply(double o) { return this.copy().multiply_s(o); }
}
