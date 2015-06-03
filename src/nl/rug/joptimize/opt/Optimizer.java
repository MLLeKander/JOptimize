package nl.rug.joptimize.opt;

public interface Optimizer {
	public OptParam optimize(SeperableCostFunction ds, OptParam init);
}
