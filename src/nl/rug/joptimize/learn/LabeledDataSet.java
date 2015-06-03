package nl.rug.joptimize.learn;

public class LabeledDataSet {
    private double[][] data;
    private int[] labels;
    
    public LabeledDataSet(double[][] data, int[] labels) {
        this.data = data;
        this.labels = labels;
    }
    
    /*public LabeledDataSet(List<? extends DataExample> examples, List<Integer> labels) {
        this.labels = new int[labels.size()];
        int i = 0;
        for (Integer label : labels) {
            this.labels[i++] = label;
        }
        
        this.data = new
    }*/
    
    public int getLabel(int ndx) {
        return labels[ndx];
    }

    public double[] getData(int ndx) {
        return data[ndx];
    }

    public double getData(int ndx, int dimension) {
        return data[ndx][dimension];
    }

    public int size() {
        return data.length;
    }

    public int dimensions() {
        // Assumes data has at least one element.
        return data[0].length;
    }
}
