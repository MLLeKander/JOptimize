package nl.rug.joptimize.opt;

import java.util.Random;

import nl.rug.joptimize.learn.Vectorlike;

public abstract class AbstractOptParam<ParamType extends OptParam<ParamType>> implements OptParam<ParamType>, Vectorlike {
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
    public ParamType sqrt() { return this.copy().sqrt_s(); }

    @Override
    public ParamType inv() { return this.copy().inv_s(); }

    @Override
    public ParamType abs() { return this.copy().abs_s(); }

    @Override
    public ParamType add(double o) { return this.copy().add_s(o); }

    @Override
    public ParamType multiply(double o) { return this.copy().multiply_s(o); }

    @Override
    public ParamType ubound(double o) { return this.copy().ubound_s(o); }

    @Override
    public ParamType lbound(double o) { return this.copy().lbound_s(o); }

    @Override
    public ParamType random(Random r) { return this.copy().random_s(r); }

    @Override    
    public abstract int length();

    @Override
    public abstract void set(int dim, double val);

    @Override
    public abstract double get(int dim);

    public static void setFlatIndex(double[][] arr, int ndx, double val) {
        int dim1 = arr.length;
        int row = ndx % dim1;
        int col = ndx / dim1;
        arr[row][col] = val;
    }
    
    public static double getFlatIndex(double[][] arr, int ndx) {
        int dim1 = arr.length;
        int row = ndx % dim1;
        int col = ndx / dim1;
        return arr[row][col];
    }
}
