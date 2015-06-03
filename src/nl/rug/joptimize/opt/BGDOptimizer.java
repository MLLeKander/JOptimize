package nl.rug.joptimize.opt;

public class BGDOptimizer extends AbstractOptimizer {
	private double learningRate;
	private double epsilon;
	private int tMax;

	public BGDOptimizer(double learningRate, double epsilon, int tMax) {
		this.learningRate = learningRate;
		this.epsilon = epsilon;
		this.tMax = tMax;
	}

	public OptParam optimize(SeperableCostFunction ds, OptParam initParams) {
		OptParam params = initParams;
		int size = ds.size();

		for (int t = 0; t < tMax && ds.error(params) < epsilon; t++) {
			OptParam nextParams = params.zero();
			for (int i = 0; i < size; i++) {
				OptParam partialGrad = ds.deriv(params, i);
				nextParams.add_s(partialGrad);
				this.notifyExample(partialGrad);
			}

			params.add_s(nextParams.multiply_s(learningRate));
			this.notifyEpoch(params);
		}
		return params;
	}
}
