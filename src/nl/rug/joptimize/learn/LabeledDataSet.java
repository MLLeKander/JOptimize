package nl.rug.joptimize.learn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Random;
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
    
    private static double[][] allocateSpace(File input) throws IOException {
        LineNumberReader in = new LineNumberReader(new BufferedReader(new FileReader(input)));
        String line = in.readLine();
        int dims = line.split(",").length-1;
        in.skip(Long.MAX_VALUE);
        int elements = in.getLineNumber();
        in.close();
        return new double[elements][dims];
    }
    
    public static LabeledDataSet parseDataFile(File input) throws IOException  {
        double[][] data = allocateSpace(input);
        int elements = data.length;
        int dims = data[0].length;
        int[] labels = new int[elements];
        
        BufferedReader in = new BufferedReader(new FileReader(input));
        try {
            for (int row = 0; row < elements; row++) {
                String line = in.readLine();
                String[] split = line.split(",");
                if (split.length != dims+1) {
                    throw new IllegalArgumentException(line+" does not contain "+dims+" elements");
                }
                for (int col = 0; col < dims; col++) {
                    data[row][col] = Double.valueOf(split[col]);
                }
                labels[row] = Integer.valueOf(split[dims]);
            }
            if (in.readLine() != null) {
                throw new IllegalArgumentException("Inappropriate number of lines");
            }
        } finally {
            in.close();
        }

        // Compensate for 1-index labels...
        boolean hasZero = false;
        for (int i : labels) {
            if (i == 0) {
                hasZero = true;
                break;
            }
        }
        if (!hasZero) {
            for (int i = 0; i < labels.length; i++) {
                labels[i]--;
            }
        }

        return new LabeledDataSet(data, labels);
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
			vars[i] = Math.sqrt(vars[i] / (size-1));
			if (vars[i] < EPS) {
				if (!discardEmpty) {
					throw new IllegalArgumentException("Feature "+i+" has (near-)zero variance.");
				} else {
					holes++;
				}
			}
		}
		
		if (holes > 0) {
			System.err.println("Removing "+holes+" holes...");
		}

		double[][] newData = zscoreCopy(holes, means, vars);
		return new LabeledDataSet(newData, this.labels);
    }
    
    private double[][] zscoreCopy(int holes, double[] means, double[] vars) {
        int dims = dimensions(), size = size();
        double[][] out = new double[size][dims-holes];
        for (int i = 0; i < size; i++) {
            int outNdx = 0;
            for (int j = 0; j < dims; j++) {
                if (vars[j] < EPS) {
                    continue;
                }

                double tmp = (data[i][j]-means[j])/vars[j];
                out[i][outNdx] = tmp;
                outNdx++;
            }
        }
        return out;
    }
    
    public SplitLabeledDataSet split(int aSize, long randSeed) {
        int dims = dimensions(), size = size();
        Random rand = new Random(randSeed);
        for (int i = 0; i < aSize; i++) {
            int sample = rand.nextInt(size - i) + i;
            if (sample != i) {
                double[] dataTmp = data[i];
                int labelTmp = labels[i];
                data[i] = data[sample];
                labels[i] = labels[sample];
                data[sample] = dataTmp;
                labels[sample] = labelTmp;
            }
        }
        double[][] dataA = new double[aSize][dims], dataB = new double[size-aSize][dims];
        for (int i = 0; i < size; i++) {
            if (i < aSize) {
                System.arraycopy(data[i], 0, dataA[i], 0, dims);
            } else {
                System.arraycopy(data[i], 0, dataB[i-aSize], 0, dims);
            }
        }
        int[] labelsA = new int[aSize], labelsB = new int[size-aSize];
        System.arraycopy(labels, 0, labelsA, 0, aSize);
        System.arraycopy(labels, aSize, labelsB, 0, size-aSize);

        return new SplitLabeledDataSet(new LabeledDataSet(dataA, labelsA), new LabeledDataSet(dataB, labelsB));
    }
}
