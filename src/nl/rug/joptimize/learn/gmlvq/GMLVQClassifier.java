package nl.rug.joptimize.learn.gmlvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GMLVQClassifier implements Classifier<GMLVQOptParam> {

    GMLVQOptParam params;
    GMLVQOptParam init;
    Optimizer<GMLVQOptParam> opt;

    public GMLVQClassifier(Optimizer<GMLVQOptParam> opt, GMLVQOptParam init) {
        this.opt = opt;
        this.init = init;
    }

    public GMLVQClassifier(GMLVQOptParam init) {
        this.init = init;
    }

    @Override
    public GMLVQOptParam train(LabeledDataSet ds) {
        return train(new GMLVQCostFunction(ds));
    }

    @Override
    public GMLVQOptParam train(SeparableCostFunction<GMLVQOptParam> cf) {
        this.params = opt.optimize(cf, this.init);
        return this.params;
    }

    @Override
    public int classify(double[] data) {
        if (params == null) {
            throw new IllegalStateException("Classifier must be trained before classifying.");
        }

        return params.getClosestProtoLabel(data);
    }

    public GMLVQOptParam getParams() {
        return this.params;
    }
}
