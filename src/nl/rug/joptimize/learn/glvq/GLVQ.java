package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.Optimizer;

public class GLVQ implements Classifier {

    GLVQOptParam params;
    GLVQOptParam init;
    Optimizer<GLVQOptParam> opt;
    public GLVQCostFunction cf;

    public GLVQ(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, GLVQOptParam init) {
        this.opt = opt;
        this.init = init;
        cf = new GLVQCostFunction(ds);
        this.train(ds);
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GLVQOptParam> opt) {
        this(ds, opt, new GLVQOptParam(ds));
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, int prototypesPerClass) {
        this(ds, opt, new GLVQOptParam(prototypesPerClass, ds));
    }

    public GLVQ(LabeledDataSet ds, Optimizer<GLVQOptParam> opt, int[] prototypesPerClass) {
        this(ds, opt, new GLVQOptParam(prototypesPerClass, ds));
    }

    public void setInit(GLVQOptParam init) {
        this.init = init;
    }

    @Override
    public void train(LabeledDataSet ds) {
        this.params = opt.optimize(new GLVQCostFunction(ds), this.init);
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
