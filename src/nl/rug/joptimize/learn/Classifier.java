package nl.rug.joptimize.learn;

public interface Classifier {
    public void train(LabeledDataSet testSet);

    public int classify(DataExample e);
}
