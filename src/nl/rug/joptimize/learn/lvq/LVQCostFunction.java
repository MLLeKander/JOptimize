package nl.rug.joptimize.learn.lvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.SeperableCostFunction;
import nl.rug.joptimize.opt.OptParam;

public class LVQCostFunction implements SeperableCostFunction {
	private final LabeledDataSet ds;
	
	public LVQCostFunction(LabeledDataSet ds) {
		this.ds = ds;
	}

	@Override
	public double error(OptParam params) {
		double out = 0;
		int size = this.size();
		for (int i = 0; i < size; i++) {
			out += error(params, i);
		}
		return out;
	}
	
	@Override
	public double error(OptParam params, int exampleNdx) {
		LVQOptParam lvqo = (LVQOptParam)params;
		double[] data = ds.getData(exampleNdx);
		int label = ds.getLabel(exampleNdx);
		// J = same, I = different
		int j = lvqo.getClosestCorrectProtoNdx(data, label), i = lvqo.getClosestIncorrectProtoNdx(data, label);
		double dj = lvqo.dist(j, data), di = lvqo.dist(i, data);
		return (di-dj)/(di+dj);
	}

	@Override
	public OptParam deriv(OptParam params) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public OptParam deriv(OptParam params, int exampleNdx) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public OptParam hesseDiag(OptParam params) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public OptParam hesseDiag(OptParam params, int exampleNdx) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int size() {
		return ds.size();
	}

}
