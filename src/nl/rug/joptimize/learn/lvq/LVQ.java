package nl.rug.joptimize.learn.lvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;

public class LVQ implements Classifier {

	LVQOptParam params;
	Optimizer opt;

	public LVQ(LabeledDataSet examples, Optimizer opt) {
		this.opt = opt;
		this.train(examples);
	}

	@Override
	public void train(LabeledDataSet examples) {
		// TODO Write cost function class
		// TODO Write LVQOptParam initialization
	}

	@Override
	public int classify(double[] data) {
		assert (params != null);

		// MIN_VALUE should never be a class label...
		return params.getClosestProtoLabel(data);
	}

}
