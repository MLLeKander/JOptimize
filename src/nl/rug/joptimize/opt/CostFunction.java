package nl.rug.joptimize.opt;

public interface CostFunction {

    public abstract double error(OptParam params);

    public abstract OptParam deriv(OptParam params);

    public abstract OptParam hesseDiag(OptParam params);

    public abstract int size();

    public abstract OptParam zero();

}