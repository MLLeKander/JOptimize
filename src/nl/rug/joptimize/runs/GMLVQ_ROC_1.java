// java -cp bin nl.rug.joptimize.runs.GMLVQ_ROC_1 ../../segment.csv --outdir out/segmentation_adam_1 --zscore false --split 2079 --splitSeed 1 --seed 1
package nl.rug.joptimize.runs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

import nl.rug.joptimize.Arguments;
import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.SplitLabeledDataSet;
import nl.rug.joptimize.learn.gmlvq.GMLVQOptParam;

public class GMLVQ_ROC_1 {
    public static class ROC_Score implements Comparable<ROC_Score> {
        double score;
        int label;
        public ROC_Score(double dist, int label) {
            this.score = dist;
            this.label = label;
        }
        
        public int compareTo(ROC_Score that) {
            return (int) Math.signum(this.score - that.score);
        }
        public String toString() {
            return this.score+" "+this.label;
        }
    }
    
    public static double[][] readTrainedBlock(BufferedReader reader) throws IOException {
        ArrayList<String[]> lines = new ArrayList<String[]>();
        String line;
        while ((line = reader.readLine()) != null && !"".equals(line.trim())) {
            lines.add(line.trim().split(" +"));
        }
        
        int rows = lines.size(), cols = lines.get(0).length;
        double[][] arr = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            String[] row = lines.get(r);
            for (int c = 0; c < cols; c++) {
                arr[r][c] = Double.parseDouble(row[c]);
            }
        }
        return arr;
    }
    
    public static GMLVQOptParam readTrained(BufferedReader trainedReader) throws IOException {
        double[][] protos = readTrainedBlock(trainedReader);
        if (protos.length != 2) {
            throw new RuntimeException("ROC must be computed on two-class problems");
        }
        double[][] weights = readTrainedBlock(trainedReader);
        int[] labels = {0,1};
        return new GMLVQOptParam(protos, weights, labels);
    }
    
    public static void main(String[] argArr) throws IOException {
        Arguments args = new Arguments(argArr);
        if (!args.hasDefault()) {
            System.err.println("No data file given.");
            return;
        }

        String outDir = args.get("outdir");
        BufferedReader trainedReader = new BufferedReader(new FileReader(new File(outDir, "trained.txt")));
        GMLVQOptParam param = readTrained(trainedReader);
        trainedReader.close();
        
        LabeledDataSet dsTmp = LabeledDataSet.parseDataFile(new File(args.getDefault()));
        if (args.getBool("zscore", false)) {
            dsTmp = dsTmp.zscore();
        }
        SplitLabeledDataSet split = dsTmp.split(args.getInt("split"),args.getLong("splitSeed"));
        LabeledDataSet dsTest = split.b;
        
        ROC_Score[] scores = new ROC_Score[dsTest.size()];
        int Z = 0, O = 0; // P and N from Fawcett (‎2006)
        
        for (int i = 0; i < scores.length; i++) {
            double[] dat = dsTest.getData(i);
            double score = param.dist(0, dat) - param.dist(1, dat);
            int label = dsTest.getLabel(i);
            scores[i] = new ROC_Score(score, label);
            
            if (label == 0) {
                Z++;
            } else {
                O++;
            }
        }
        
        Arrays.sort(scores);
        
        // Only create file when we actually use it.
        PrintStream rocFile = new PrintStream(new File(outDir, "roc.txt"));
        
        double prevScore = Double.NEGATIVE_INFINITY;
        int FZ = 0, TZ = 0; // FP and TP from Fawcett (2006)
        
        for (int i = 0; i < scores.length; i++) {
            double score = scores[i].score;
            int label = scores[i].label;
            if (score != prevScore) {
                rocFile.printf("%.4f %.4f\n",FZ/(double)O,TZ/(double)Z);
                prevScore = score;
            }
            if (label == 0) {
                TZ++;
            } else {
                FZ++;
            }
        }
        
        rocFile.close();
    }
}
