// java -cp bin nl.rug.joptimize.LVQMain segment.dat  --rate 0.1 --epsilon 1e-4 --tmax 1000000 --freq 500 --opt bgd
package nl.rug.joptimize;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.grlvq.GRLVQ;
import nl.rug.joptimize.learn.grlvq.GRLVQOptParam;
import nl.rug.joptimize.opt.AbstractOptimizer;
import nl.rug.joptimize.opt.OptObserver;
import nl.rug.joptimize.opt.observers.CountObserver;
import nl.rug.joptimize.opt.observers.TimeObserver;
import nl.rug.joptimize.opt.optimizers.BGD;
import nl.rug.joptimize.opt.optimizers.SGD;
import nl.rug.joptimize.opt.optimizers.WA_BGD;
import nl.rug.joptimize.opt.optimizers.vSGD;

public class LVQMain {
    public static final AbstractOptimizer<GRLVQOptParam> DEFAULT_OPT = new BGD<>(0.1, 1e-5, 10000);
    
    public static Map<String, String> parseArgs(String[] args) {
        Map<String, String> out = new HashMap<String, String>();
        StringBuilder sb = new StringBuilder();
        String sep = "";

        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                assert (args.length > i + 1);
                out.put(args[i].toLowerCase(), args[i + 1]);
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
    
    public static AbstractOptimizer<GRLVQOptParam> createOptimizer(Map<String, String> args) {
        if (!args.containsKey("--opt")) {
            return DEFAULT_OPT;
        }
        String opt = args.get("--opt").toUpperCase().replaceAll("[\\p{Punct} ]+", "");
        if (opt.equals("BGD")) {
            return new BGD<>(args);
        } else if (opt.equals("SGD")) {
            return new SGD<>(args);
        } else if (opt.equals("WABGD")) {
            return new WA_BGD<>(args);
        } else if (opt.equals("VSGD")) {
            return new vSGD<>(args);
        }
        throw new IllegalArgumentException("Unknown optimizer: "+opt);
    }

    public static void main(String[] args) throws FileNotFoundException {
        Map<String, String> argMap = parseArgs(args);
        if (!argMap.containsKey("")) {
            System.err.println("No data file given.");
            printUsage();
        }
        final LabeledDataSet ds = LabeledDataSet.parseDataFile(new File(argMap.get("")));
        AbstractOptimizer<GRLVQOptParam> opt = createOptimizer(argMap);
        System.out.println("Proceeding with optimizer: "+opt.getClass().getName());
        
        final int freq = argMap.containsKey("--freq") ? Integer.parseInt(argMap.get("--freq")) : 50;
        opt.addObs(new OptObserver<GRLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GRLVQOptParam params, double error) {
                if (t++%freq == 0) {
                    //System.out.println(params);
                    int err = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            err++;
                        }
                    }
                    System.out.println(t+": "+err+" / "+ds.size()+", "+error);//+ " | "+params);
                }
            }

            @Override
            public void notifyExample(GRLVQOptParam params) { }
        });
        
        CountObserver<GRLVQOptParam> counter = new CountObserver<>();
        TimeObserver<GRLVQOptParam> timer = new TimeObserver<>();
        opt.addObs(counter);
        opt.addObs(timer);

        GRLVQOptParam p = new GRLVQOptParam(new double[][]{{0,0.5},{0,-0.5}}, new double[]{1,1}, new int[]{0,1});
        GRLVQ lvq = new GRLVQ(ds, opt, p);
        
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        System.out.println(err+" / "+ds.size());
        System.out.println(counter.getEpochCount());
        
        System.out.println(timer.end - timer.start + " nanoseconds");
        System.out.println(lvq.getParams());
    }

}
