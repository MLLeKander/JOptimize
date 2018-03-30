package nl.rug.joptimize.learn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import nl.rug.joptimize.opt.ClassificationOptParam;

public class ROC {
    public static class ROCPoint {
        double fpr, tpr;
        public ROCPoint(double fpr_, double tpr_) {
            fpr = fpr_;
            tpr = tpr_;
        }
        
        public String toString() {
            return String.format("(%.3f,%.3f)",this.tpr,this.fpr);
        }
    }
    
    public static class ROCScore implements Comparable<ROCScore> {
        double score;
        int label;
        public ROCScore(double dist, int label) {
            this.score = dist;
            this.label = label;
        }
        
        public int compareTo(ROCScore that) {
            return (int) Math.signum(this.score - that.score);
        }
        
        public String toString() {
            return this.score+" "+this.label;
        }
    }
    
    public static <ParamType extends ClassificationOptParam<ParamType>>
            double auc(ParamType p, LabeledDataSet ds) {
        return auc(rocPoints(p, ds));
    }
    
    public static double auc(List<ROCPoint> points) {
        double area = 0;
        for (int i = 0; i < points.size()-1; i++) {
            area += trapezoidArea(points.get(i), points.get(i+1));
        }
        return area;
    }
    
    private static double trapezoidArea(ROCPoint a, ROCPoint b) {
        return Math.abs(a.fpr-b.fpr)*(a.tpr+b.tpr)/2;
    }
    
    public static <ParamType extends ClassificationOptParam<ParamType>>
            List<ROCPoint> rocPoints(ParamType p, LabeledDataSet ds) {
        return rocPoints(p.rocScores(ds), ds);
    }
    
    public static List<ROCPoint> rocPoints(double[] scores, LabeledDataSet ds) {
        ROCScore[] rocScores = new ROCScore[ds.size()];

        int Z = 0, O = 0; // P and N from Fawcett (‎2006)

        for (int i = 0; i < ds.size(); i++) {
            int label = ds.getLabel(i);
            rocScores[i] = new ROCScore(scores[i], label);
            
            if (label == 0) {
                Z++;
            } else {
                O++;
            }    
        }
        
        Arrays.sort(rocScores);
        
        double prevScore = Double.NEGATIVE_INFINITY;
        int FZ = 0, TZ = 0; // FP and TP from Fawcett (2006)
        ArrayList<ROCPoint> out = new ArrayList<ROCPoint>();
        
        for (int i = 0; i < rocScores.length; i++) {
            double score = rocScores[i].score;
            int label = rocScores[i].label;
            if (score != prevScore) {
                out.add(new ROCPoint(FZ/(double)O,TZ/(double)Z));
                prevScore = score;
            }
            if (label == 0) {
                TZ++;
            } else {
                FZ++;
            }
        }
        out.add(new ROCPoint(1,1));
        
        return out;
    }
}
