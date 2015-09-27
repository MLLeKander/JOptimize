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
    private final static double EPS = 1e-10;

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
    
    public double[][] averageProtos() {
        int classes = this.classes(), dims = this.dimensions(), size = this.size();
        double[][] protos = new double[classes][dims];
        int[] counts = new int[classes];
        for (int i = 0; i < size; i++) {
            double[] row = this.getData(i);
            int label = this.getLabel(i);
            counts[label]++;
            for (int j = 0; j < dims; j++) {
                protos[label][j] += row[j];
            }
        }
        
        for (int i = 0; i < classes; i++) {
            for (int j = 0; j < dims; j++) {
                protos[i][j] /= counts[i];
            }
        }
        return protos;
    }
    
    public int[] labels() {
    	int[] out = new int[classes()];
    	for (int i = 0; i < classes(); i++) {
    		out[i] = i;
    	}
    	return out;
    }
    
    public LabeledDataSet zscore() {
    	return zscore(true);
    }
        
    public LabeledDataSet zscore(boolean discardEmpty) {
	    int dims = this.dimensions(), size = this.size();
		double[] means = new double[dims];
		for (double[] row : data) {
			for (int i = 0; i < dims; i++) {
				means[i] += row[i];
			}
		}
		for (int i = 0; i < dims; i++) {
			means[i] /= size;
		}
		
		double[] vars = new double[dims];
		for (double[] row : data) {
			for (int i = 0; i < dims; i++) {
				double tmp = row[i] - means[i];
				vars[i] += tmp*tmp;
			}
		}
		
		int holes = 0;
		for (int i = 0; i < dims; i++) {
			vars[i] = Math.sqrt(vars[i]);
			if (vars[i] < EPS && !discardEmpty) {
				throw new IllegalArgumentException("Feature "+i+" has (near-)zero variance.");
			}
			holes++;
		}
		
		if (holes > 0) {
			System.err.println("Removing "+holes+" holes.");
		}
		
		double[][] newData = zscoreCopy(holes, means, vars);
		return new LabeledDataSet(newData, this.labels);
    }
    
    private double[][] zscoreCopy(int holes, double[] means, double[] vars) {
	    int dims = this.dimensions(), size = this.size();
    	double[][] out = new double[size-holes][dims];
    	for (int i = 0; i < size; i++) {
    		int outNdx = 0;
    		for (int j = 0; j < dims; j++) {
    			if (vars[j] < EPS) {
    				continue;
    			}
    			
    			out[i][outNdx] = (data[i][j]-means[j])/vars[j];
    			outNdx++;
    		}
    	}
    	return out;
    }
    
}
