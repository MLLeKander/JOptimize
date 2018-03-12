package nl.rug.joptimize.opt;

import nl.rug.joptimize.learn.LabeledDataSet;

public interface ClassificationOptParam<ParamType extends ClassificationOptParam<ParamType>> extends OptParam<ParamType> {
    public int classify(double[] data);

    public double[] rocScores(LabeledDataSet ds);
}
