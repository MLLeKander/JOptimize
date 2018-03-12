package nl.rug.joptimize.learn;

import java.util.ArrayList;
import java.util.List;

import nl.rug.joptimize.learn.ROC.ROCPoint;

public class ROCTest {
    public static void main(String[] args) {
        List<ROCPoint> lst = new ArrayList<ROCPoint>();

        lst.add(new ROCPoint(0, 0));
        lst.add(new ROCPoint(0.25, 0.75));
        lst.add(new ROCPoint(1, 1));
        System.out.println(lst);
        
        System.out.println(ROC.auc(lst));
    }
}