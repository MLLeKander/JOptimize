package nl.rug.joptimize.learn.simple;

import java.util.Random;

import nl.rug.joptimize.opt.AbstractOptParam;

public class SingleVarOptParam extends AbstractOptParam<SingleVarOptParam> {
    public double x;
    
    public SingleVarOptParam() {
        this.x = 0;
    }
    
    public SingleVarOptParam(double x) {
        this.x = x;
    }

    @Override
    public SingleVarOptParam add_s(SingleVarOptParam o) {
        x += o.x;
        return this;
    }

    @Override
    public SingleVarOptParam sub_s(SingleVarOptParam o) {
        x -= o.x;
        return this;
    }

    @Override
    public SingleVarOptParam zero_s() {
        x = 0;
        return this;
    }

    @Override
    public SingleVarOptParam one_s() {
        x = 1;
        return this;
    }

    @Override
    public SingleVarOptParam dotprod_s(SingleVarOptParam o) {
        x *= o.x;
        return this;
    }

    @Override
    public SingleVarOptParam sqrt_s() {
        x = Math.sqrt(x);
        return this;
    }

    @Override
    public SingleVarOptParam inv_s() {
        x = 1/x;
        return this;
    }

    @Override
    public SingleVarOptParam abs_s() {
        if (x < 0) {
            x = -x;
        }
        return this;
    }

    @Override
    public SingleVarOptParam lbound_s(double o) {
        if (x < o) {
            x = o;
        }
        return this;
    }

    @Override
    public SingleVarOptParam ubound_s(double o) {
        if (x < o) {
            x = o;
        }
        return this;
    }

    @Override
    public SingleVarOptParam add_s(double o) {
        x += o;
        return this;
    }

    @Override
    public SingleVarOptParam multiply_s(double o) {
        x *= o;
        return this;
    }

    @Override
    public SingleVarOptParam copy() {
        return new SingleVarOptParam(x);
    }

    @Override
    public double squaredNorm() {
        return x*x;
    }

    @Override
    public SingleVarOptParam random_s(Random r) {
        x = r.nextDouble();
        return this;
    }

    @Override
    public int length() {
        return 1;
    }

    @Override
    public void set(int dim, double val) {
        x = val;
    }

    @Override
    public double get(int dim) {
        return x;
    }
}
