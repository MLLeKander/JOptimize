package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GLVQClassifier implements Classifier<GLVQOptParam> {

    GLVQOptParam params;
    GLVQOptParam init;
    Optimizer<GLVQOptParam> opt;

    public GLVQClassifier(Optimizer<GLVQOptParam> opt, GLVQOptParam init) {
        this.opt = opt;
        this.init = init;
    }

    public void setInit(GLVQOptParam init) {
        this.init = init;
    }

    @Override
    public GLVQOptParam train(LabeledDataSet ds) {
        return train(new GLVQCostFunction(ds));
    }

    @Override
    public GLVQOptParam train(SeparableCostFunction<GLVQOptParam> cf) {
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

    public GLVQOptParam getParams() {
        return this.params;
    }
}
