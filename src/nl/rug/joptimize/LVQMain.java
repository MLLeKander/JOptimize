package nl.rug.joptimize;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.learn.glvq.*;
import nl.rug.joptimize.opt.*;
import nl.rug.joptimize.opt.observers.*;
import nl.rug.joptimize.opt.optimizers.*;

public class LVQMain {
    public static final AbstractOptimizer<GLVQOptParam> DEFAULT_OPT = new BGD<>(0.1, 1e-5, 10000);
    
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
    
    public static AbstractOptimizer<GLVQOptParam> createOptimizer(Map<String, String> args) {
        if (!args.containsKey("--opt")) {
            return DEFAULT_OPT;
        }
        String opt = args.get("--opt").toUpperCase().replaceAll("[\\p{Punct} ]+", "");
        if (opt.equals("BGD")) {
            return new BGD<>(args);
        } else if (opt.equals("WABGD")) {
            return new WA_BGD<>(args);
        } else if (opt.equals("VSGD")) {
            return new vSGD<>(args);
        }
        return null;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Map<String, String> argMap = parseArgs(args);//new String[]{"/home/michael/Documents/Internship/segment.dat"});
        if (!argMap.containsKey("")) {
            System.err.println("No data file given.");
            printUsage();
        }
        if (argMap.containsKey("--opt")) {
            
        }
        final LabeledDataSet ds = LabeledDataSet.parseDataFile(new File(argMap.get("")));
        //BGD<GRLVQOptParam> opt = new BGD<>(0.1, 1e-5, 100000);
        //WA_BGD<GLVQOptParam> opt = new WA_BGD<>(0.1, 1e-3, 100000, 10, .75, 1.5);
        AbstractOptimizer<GLVQOptParam> opt = createOptimizer(argMap);
        System.out.println("Proceeding with optimizer: "+opt);

        CountObserver<GLVQOptParam> counter = new CountObserver<>();
        TimeObserver<GLVQOptParam> timer = new TimeObserver<>();
        opt.addObs(new OptObserver<GLVQOptParam>() {
            int t = 0;
            @Override
            public void notifyEpoch(GLVQOptParam params, double error) {
                if (t++%5 == 0) {
                    System.out.println(params);
                    int err = 0;
                    for (int i = 0; i < ds.size(); i++) {
                        if (params.getClosestProtoLabel(ds.getData(i)) != ds.getLabel(i)) {
                            err++;
                        }
                    }
                    System.out.println(err+" / "+ds.size()+", "+error);//+ " | "+params);
                }
            }

            @Override
            public void notifyExample(GLVQOptParam params) { }
        });
        opt.addObs(counter);
        opt.addObs(timer);

        GLVQ lvq = new GLVQ(ds, opt);
        int err = 0;
        for (int i = 0; i < ds.size(); i++) {
            if (lvq.classify(ds.getData(i)) != ds.getLabel(i)) {
                err++;
            }
        }
        System.out.println(err+" / "+ds.size());
        System.out.println(counter.getEpochCount());
        
        System.out.println(timer.end - timer.start + " nanoseconds");
    }

}
