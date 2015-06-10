package nl.rug.joptimize.learn;

import java.util.TreeSet;

public class LabeledDataSet {
    private double[][] data;
    private int[] labels;
    private int maxLabel;

    public LabeledDataSet(double[][] data, int[] labels) {
        this.data = data;
        this.labels = labels;

        TreeSet<Integer> set = new TreeSet<>();
        for (int i : labels) {
            set.add(i);
        }

        this.maxLabel = set.last();

        assert (set.first() == 0);
        for (int i = maxLabel; i > 0; i--) {
            assert (set.contains(i));
        }
    }

    public int getLabel(int ndx) {
        return this.labels[ndx];
    }

    public double[] getData(int ndx) {
        return this.data[ndx];
    }

    public double getData(int ndx, int dimension) {
        return this.data[ndx][dimension];
    }

    public int size() {
        return this.data.length;
    }

    public int dimensions() {
        // Assumes data has at least one element.
        return this.data[0].length;
    }

    public int classes() {
        return this.maxLabel+1;
    }
}
