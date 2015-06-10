package nl.rug.joptimize;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.glvq.GLVQ;
import nl.rug.joptimize.learn.glvq.GRLVQOptParam;
import nl.rug.joptimize.opt.BGDOptimizer;
import nl.rug.joptimize.opt.CountOptObserver;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.TimeOptObserver;
public class LVQMain {
    public static Map<String, String> parseArgs(String[] args) {
        Map<String, String> out = new HashMap<String, String>();
        StringBuilder sb = new StringBuilder();
        String sep = "";

        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                assert (args.length > i + 1);
                out.put(args[i], args[i + 1]);
                i++;
            } else {
                sb.append(sep);
                sb.append(args[i]);
                sep = " ";
            }
        }
        if (sb.length() > 0) {
            out.put("", sb.toString());
        }
        return out;
    }

    public static void printUsage() {
        String className = LVQMain.class.getName();
        System.out.println("java " + className
                + " dataFile [--numProtos numProtos] [--protoCounts proto1,proto2] [--opt optName]");
        System.exit(-1);
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

    public static void main(String[] args) throws FileNotFoundException {
        Map<String, String> params = parseArgs(new String[]{"/home/michael/Documents/Internship/segment.dat"});
        if (!params.containsKey("")) {
            System.err.println("No data file given.");
            printUsage();
        }
        final LabeledDataSet ds = parseDataFile(new File(params.get("")));
        BGDOptimizer<GRLVQOptParam> opt = new BGDOptimizer<>(0.1, 1e-5, 100000);

        CountOptObserver<GRLVQOptParam> counter = new CountOptObserver<>();
        TimeOptObserver<GRLVQOptParam> timer = new TimeOptObserver<>();
        opt.addObs(new OptObserver<GRLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GRLVQOptParam params, double error) {
                if (t++%500 == 0) {
                    int err = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            err++;
                        }
                    }
                    System.out.println(err+" / "+ds.size()+", "+error);
                }
            }

            @Override
            public void notifyExample(GRLVQOptParam params) { }
        });
        // opt.addObs(new PrintOptObserver<LVQOptParam>());
        opt.addObs(counter);
        opt.addObs(timer);

        GLVQ lvq = new GLVQ(ds, opt);
        //System.out.println(lvq.getParams());
        //System.out.println(lvq.classify(new double[] { -1, 0 }));
        //System.out.println(lvq.classify(new double[] { -.01, 0 }));
        //System.out.println(lvq.classify(new double[] { 0, 1 }));
        //System.out.println(lvq.classify(new double[] { 0, -.01 }));
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        System.out.println(err+" / "+ds.size());
        System.out.println(counter.getEpochCount());
        System.out.println(counter.getExampleCount());
        
        System.out.println(timer.end - timer.start + " nanoseconds");
    }

}
