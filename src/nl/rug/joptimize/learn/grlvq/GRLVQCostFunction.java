package nl.rug.joptimize.learn.grlvq;

import java.util.Arrays;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.SeparableCostFunction;

public class GRLVQCostFunction implements SeparableCostFunction<GRLVQOptParam> {
    private final LabeledDataSet ds;

    public GRLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
    }

    @Override
    public double error(GRLVQOptParam params) {
        // TODO Vectorized?
        double out = 0;
        int size = this.size();
        for (int i = 0; i < size; i++) {
            out += error(params, i);
        }
        return out;
    }

    @Override
    public double error(GRLVQOptParam params, int exampleNdx) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);
        double dj = params.dist(j, data), dk = params.dist(k, data);
        return (dj - dk) / (dk + dj);
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params) {
        // TODO Vectorized?
        GRLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            deriv(params, i, out);
        }
        return out;
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params, int exampleNdx) {
        GRLVQOptParam out =  deriv(params, exampleNdx, params.zero());
//        System.out.println("deriv: "+exampleNdx+" "+Arrays.toString(ds.getData(exampleNdx)));
        return out;
    }

    @Override
    public GRLVQOptParam deriv(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
        // (2*dk*dj' - 2*dj*dk')/(dj+dk)^2
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        if (j < 0 || k < 0) System.out.println("WTF is happening? "+params);
        
        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);
        double dSum = dj+dk;

        double tmp = 2 / (dSum * dSum);
        // TODO Check signs.
        double tmpJ = tmp * 2 * dk, tmpK = -tmp * 2 * dj;
        double tmpWj = tmp * dk, tmpWk = -tmp * dj; 
        int dims = params.dimensions();

        double djP, dkP, sav;
        for (int ndx = 0; ndx < dims; ndx++) {
            double diffJ = pj[ndx] - data[ndx];
            double diffK = pk[ndx] - data[ndx];
                        
            djP = 2*weights[ndx]*diffJ; dkP = 0;
            sav = tmpJ * diffJ * weights[ndx];
            opj[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP),sav, "opj'");
            
            djP = 0; dkP = 2*weights[ndx]*diffK;
            sav = tmpK * diffK * weights[ndx];
            opk[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP), sav, "opk'");
            
            // (2*dk*(pj-data)^2 - 2*dj*(pk-data)^2)/(dj+dk)^2
            djP = weights[ndx] * diffJ * diffJ; dkP = weights[ndx] * diffK * diffK;
            sav = weights[ndx] * (tmpWj * diffJ * diffJ + tmpWk * diffK * diffK);
            oweights[ndx] += sav;
            check(firstDerivLVQ(dj, dk, djP, dkP),sav, "oweights'");
        }

        return out;
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params) {
        // TODO Vectorized?
        GRLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            hesseDiag(params, i, out);
        }
        return out;
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params, int exampleNdx) {
        return hesseDiag(params, exampleNdx, params.zero());
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
        // (dj''-dk'')/(dj+dk) - 2(dj'^2-dk'^2)/(dk+dj)^2 + (dj-dk)(2(dj'-dk')^2/(dj+dk)^3 - (dj''+dk'')/(dj+dk)^2)
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);
//        double dSum = dj + dk;

//        double tmp = 4 / (dSum * dSum * dSum);
        // TODO Check signs.
//        double tmpJ = tmp * dk * dSum, tmpJ_ = tmp * dk * -2;
//        double tmpK = -tmp * dj * dSum, tmpK_ = tmp * dj * 2;
        int dims = params.dimensions();

//        double sav;
        double djP, dkP, djPP, dkPP;
        for (int ndx = 0; ndx < dims; ndx++) {
            //TODO Appropriate for weights
        	double w = weights[ndx], wSqr = w*w;
            double diffJ = pj[ndx] - data[ndx], diffK = pk[ndx] - data[ndx];
            
            djP = 2*wSqr*diffJ; dkP = 0;
            djPP = 2*wSqr; dkPP = 0;
            // TODO: Optimize this.
            opj[ndx] += secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP);
//            sav = wSqr * (tmpJ + tmpJ_ * diffJ);
//            opk[ndx] += sav;
//            check(secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP), sav, "opj''");

            djP = 0; dkP = 2*wSqr*diffJ;
            djPP = 0; dkPP = 2*wSqr;
            // TODO: Optimize this.
            opk[ndx] += secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP);
//            sav = wSqr * (tmpK + tmpK_ * diffK);
//            opk[ndx] += sav;
//            check(secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP), sav, "opk''");

            djP = 2*w*diffJ*diffJ; dkP = 2*w*diffK*diffK;
            djPP = 2*diffJ*diffJ; dkPP = 2*diffK*diffK;
            // TODO: Optimize this.
            oweights[ndx] += secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP);
//            double sav = tmp/2*((djP-dkP)*(djP-dkP)*(dj-dk) - (dj+dk)*(djP*djP - dkP*dkP));
//            oweights[ndx] += sav;
//            check(secondDerivLVQ(dj, dk, djP, dkP, djPP, dkPP), sav, "oweights''");
        }

        return out;
    }

    public void check(double a, double b, String str) {
//    	System.out.printf("??? %f %f %s %f\n", a, b, str, Math.abs((a-b) / Math.min(a,b)));
        if (Math.abs((a-b) / Math.min(a,b)) > 1e-4 && Math.abs(Math.max(a, b)) > 1e-10) {
            System.out.println("SOMETHING WRONG!!! "+str+" "+a+" "+b);
            //throw new IllegalStateException();
        } else {
            //System.out.println("Here");
        }
    }
    
    private double firstDerivLVQ(double dj, double dk, double djP, double dkP) {
        double dSum = dj+dk;
        return 2*(dk*djP - dj*dkP)/(dSum*dSum);
    }
    
    private double secondDerivLVQ(double dj, double dk, double djP, double dkP, double djPP, double dkPP) {
        double dSum = dj+dk;
        return (djPP-dkPP)/(dj+dk) - 2*(djP*djP-dkP*dkP)/(dSum*dSum) + (dj-dk)*(2*(djP-dkP)*(djP-dkP)/(dSum*dSum*dSum) - (djPP+dkPP)/(dSum*dSum));
    }

    @Override
    public int size() {
        return ds.size();
    }
}
