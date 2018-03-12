package nl.rug.joptimize.learn;

import nl.rug.joptimize.opt.OptParam;
import nl.rug.joptimize.opt.SeparableCostFunction;

public interface Classifier<ParamType extends OptParam<ParamType>> {
    public ParamType train(LabeledDataSet testSet);
    public ParamType train(SeparableCostFunction<ParamType> testSet);

    public int classify(double[] e);
}
