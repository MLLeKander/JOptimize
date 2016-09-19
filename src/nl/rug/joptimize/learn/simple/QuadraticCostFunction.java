package nl.rug.joptimize.learn.simple;

import nl.rug.joptimize.opt.CostFunction;

public class QuadraticCostFunction implements CostFunction<SingleVarOptParam> {
    @Override
    public double error(SingleVarOptParam params) {
        // TODO Auto-generated method stub
        return params.x*params.x;
    }

    @Override
    public SingleVarOptParam deriv(SingleVarOptParam params) {
        return new SingleVarOptParam(2*params.x);
    }

    @Override
    public SingleVarOptParam hesseDiag(SingleVarOptParam params) {
        return new SingleVarOptParam(2);
    }

}
