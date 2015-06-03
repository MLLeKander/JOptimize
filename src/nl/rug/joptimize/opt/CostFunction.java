package nl.rug.joptimize.opt;

public interface CostFunction {

	public double error(OptParam params);

	public OptParam deriv(OptParam params);

	public OptParam hesseDiag(OptParam params);

}