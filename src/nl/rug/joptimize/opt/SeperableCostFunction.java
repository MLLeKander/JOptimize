package nl.rug.joptimize.opt;

public interface SeperableCostFunction extends CostFunction {
    public double error(OptParam params, int exampleNdx);

    public OptParam deriv(OptParam params, int exampleNdx);

    public OptParam hesseDiag(OptParam params, int exampleNdx);
}
