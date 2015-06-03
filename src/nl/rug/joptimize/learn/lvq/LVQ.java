package nl.rug.joptimize.learn.lvq;

import nl.rug.joptimize.learn.Classifier;
import nl.rug.joptimize.learn.DataExample;
import nl.rug.joptimize.learn.LabeledDataSet;

public class LVQ implements Classifier {

    LVQOptParam params;

    public LVQ(LabeledDataSet examples) {
        this.train(examples);
    }

    @Override
    public void train(LabeledDataSet examples) {
        // TODO Auto-generated method stub

    }

    @Override
    public int classify(DataExample e) {
        // TODO Auto-generated method stub
        assert (params != null);
        return 0;
    }

}
