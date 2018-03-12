package nl.rug.joptimize.learn.grlvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GRLVQClassifier implements Classifier<GRLVQOptParam> {

    GRLVQOptParam params;
    GRLVQOptParam init;
    Optimizer<GRLVQOptParam> opt;

    public GRLVQClassifier(Optimizer<GRLVQOptParam> opt, GRLVQOptParam init) {
        this.opt = opt;
        this.init = init;
    }

    public void setInit(GRLVQOptParam init) {
        this.init = init;
    }

    @Override
    public GRLVQOptParam train(LabeledDataSet ds) {
        return train(new GRLVQCostFunction(ds));
    }

    @Override
    public GRLVQOptParam train(SeparableCostFunction<GRLVQOptParam> cf) {
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

    public GRLVQOptParam getParams() {
        return this.params;
    }
}
