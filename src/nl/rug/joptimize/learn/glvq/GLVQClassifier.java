package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GLVQClassifier implements Classifier {

    GLVQOptParam params;
    GLVQOptParam init;
    Optimizer<GLVQOptParam> opt;
    public GLVQCostFunction cf;

    public GLVQClassifier(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, GLVQOptParam init) {
        this.opt = opt;
        this.init = init;
        this.cf = new GLVQCostFunction(ds);
        this.train(ds);
    }

    public GLVQClassifier(LabeledDataSet ds, Optimizer<GLVQOptParam> opt) {
        this(ds, opt, new GLVQOptParam(ds));
    }

    public GLVQClassifier(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, int prototypesPerClass) {
        this(ds, opt, new GLVQOptParam(prototypesPerClass, ds));
    }

    public GLVQClassifier(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, int[] prototypesPerClass) {
        this(ds, opt, new GLVQOptParam(prototypesPerClass, ds));
    }

    public void setInit(GLVQOptParam init) {
        this.init = init;
    }

    @Override
    public void train(LabeledDataSet ds) {
        train(new GLVQCostFunction(ds));
    }
    
    public void train(SeparableCostFunction<GLVQOptParam> cf) {
        this.params = opt.optimize(cf, this.init);
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
