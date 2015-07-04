package nl.rug.joptimize.opt;

public abstract class AbstractOptParam<ParamType extends OptParam<ParamType>> implements OptParam<ParamType> {
    @Override
	public ParamType add(ParamType o) { return this.copy().add_s(o); }

    @Override
    public ParamType sub(ParamType o) { return this.copy().sub_s(o); }

    @Override
    public ParamType zero() { return this.copy().zero_s(); }

    @Override
    public ParamType one() { return this.copy().one_s(); }

    @Override
    public ParamType dotprod(ParamType o) { return this.copy().dotprod_s(o); }

    @Override
    public ParamType inv() { return this.copy().inv_s(); }

    @Override
    public ParamType abs() { return this.copy().abs_s(); }

    @Override
    public ParamType multiply(double o) { return this.copy().multiply_s(o); }

    @Override
    public ParamType ubound(double o) { return this.copy().ubound_s(o); }

    @Override
    public ParamType lbound(double o) { return this.copy().lbound_s(o); }
}
