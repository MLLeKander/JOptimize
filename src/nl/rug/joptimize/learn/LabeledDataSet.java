package nl.rug.joptimize.learn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
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
    public static LabeledDataSet parseDataFile(File input) throws FileNotFoundException {
        Scanner file = new Scanner(input);
        @SuppressWarnings("resource")
        Scanner line = new Scanner(file.nextLine());
        line.useDelimiter(",");

        ArrayList<Double> firstLine = new ArrayList<>();
        while (line.hasNextDouble()) {
            firstLine.add(line.nextDouble());
        }

        int dims = firstLine.size() - 1;
        ArrayList<Integer> labels = new ArrayList<>();
        labels.add((int) (double) firstLine.remove(firstLine.size() - 1));
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        data.add(firstLine);

        while (file.hasNextLine()) {
            line = new Scanner(file.nextLine());
            line.useDelimiter(",");
            if (!line.hasNext()) continue;
            ArrayList<Double> nextLine = new ArrayList<>(dims);
            for (int i = 0; i < dims; i++) {
                nextLine.add(line.nextDouble());
            }
            data.add(nextLine);
            labels.add(line.nextInt());
            assert (!line.hasNext());
        }

        int[] labelsArr = new int[labels.size()];
        for (int i = 0; i < labelsArr.length; i++) {
            labelsArr[i] = labels.get(i);
        }

        double[][] dataArr = new double[data.size()][dims];
        for (int i = 0; i < dataArr.length; i++) {
            ArrayList<Double> row = data.get(i);
            assert (row.size() == dims);
            for (int j = 0; j < dims; j++) {
                dataArr[i][j] = row.get(j);
            }
        }
        file.close();

        return new LabeledDataSet(dataArr, labelsArr);
    }
}
