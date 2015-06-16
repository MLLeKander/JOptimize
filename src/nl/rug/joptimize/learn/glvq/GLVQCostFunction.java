package nl.rug.joptimize.learn.glvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.SeperableCostFunction;

public class GLVQCostFunction implements SeperableCostFunction<GLVQOptParam> {
    private final LabeledDataSet ds;

    public GLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
    }

    @Override
    public double error(GLVQOptParam params) {
        // TODO Vectorized?
        double out = 0;
        int size = this.size();
        for (int i = 0; i < size; i++) {
            out += error(params, i);
        }
        return out;
    }

    @Override
    public double error(GLVQOptParam params, int exampleNdx) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);
        double dj = params.dist(j, data), dk = params.dist(k, data);
        return (dj - dk) / (dk + dj);
    }

    @Override
    public GLVQOptParam deriv(GLVQOptParam params) {
        // TODO Vectorized?
        GLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            deriv(params, i, out);
        }
        return out;
    }

    @Override
    public GLVQOptParam deriv(GLVQOptParam params, int exampleNdx) {
        return deriv(params, exampleNdx, params.zero());
    }

    @Override
    public GLVQOptParam deriv(GLVQOptParam params, int exampleNdx, GLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double dj = params.dist(j, data), dk = params.dist(k, data);
        double dSum = dj+dk;

        double tmp = 4 / (dSum*dSum);
        double tmpJ = tmp * dk, tmpK = -tmp * dj;
        int dims = params.dimensions();

        for (int ndx = 0; ndx < dims; ndx++) {
            opj[ndx] += tmpJ * (pj[ndx] - data[ndx]);
            //double sav = tmpJ * (pj[ndx] - data[ndx]);
            //check(firstDerivLVQ(dj, dk, 2*(pj[ndx]-data[ndx]), 0),sav);

            opk[ndx] += tmpK * (pk[ndx] - data[ndx]);
            //sav = tmpK * (pk[ndx] - data[ndx]);
            //check(firstDerivLVQ(dj, dk, 0, 2*(pk[ndx]-data[ndx])),sav);
        }

        return out;
    }

    @Override
    public GLVQOptParam hesseDiag(GLVQOptParam params) {
        // TODO Vectorized?
        GLVQOptParam out = params.zero();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            hesseDiag(params, i, out);
        }
        return out;
    }

    @Override
    public GLVQOptParam hesseDiag(GLVQOptParam params, int exampleNdx) {
        return hesseDiag(params, exampleNdx, params.zero());
    }

    @Override
    public GLVQOptParam hesseDiag(GLVQOptParam params, int exampleNdx, GLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double dj = params.dist(j, data), dk = params.dist(k, data);
        double dSum = dj + dk;

        double tmp = 4 / (dSum * dSum * dSum);
        // TODO Check signs.
        double tmpJ = tmp * dk * dSum, tmpJ_ = -tmp * dk;
        double tmpK = -tmp * dj * dSum, tmpK_ = tmp * dj;
        int dims = params.dimensions();

        //double save;
        for (int ndx = 0; ndx < dims; ndx++) {
            double diffJ = pj[ndx] - data[ndx], diffK = pk[ndx] - data[ndx];
            opj[ndx] += tmpJ + tmpJ_ * 4 * diffJ * diffJ;
            //save = tmpJ + tmpJ_ * 4 * diffJ * diffJ;
            //check(secondDerivLVQ(dj, dk, 2*(pj[ndx]-data[ndx]), 0, 2, 0), save);
            
            opk[ndx] += tmpK + tmpK_ * 4 * diffK * diffK;
            //save = tmpK + tmpK_ * 4 * diffK * diffK;
            //check(secondDerivLVQ(dj, dk, 0, 2*(pk[ndx]-data[ndx]), 0, 2), save);
        }

        return out;
    }
    
    public void check(double a, double b) {
        if (Math.abs(a-b) / Math.min(a,b) > 1e-3) {
            System.out.println(a+" "+b);
            //throw new IllegalStateException();
        } else {
            //System.out.println("Here");
        }
    }
    
    @SuppressWarnings("unused")
    private double firstDerivLVQ(double dj, double dk, double djP, double dkP) {
        double dSum = dj+dk;
        return 2*(dk*djP - dj*dkP)/(dSum*dSum);
    }

    @SuppressWarnings("unused")
    private double secondDerivLVQ(double dj, double dk, double djP, double dkP, double djPP, double dkPP) {
        double dSum = dj+dk;
        //return (djPP-dkPP)/(dj+dk) - 2*(djP*djP-dkP*dkP)/(dSum*dSum) + (dj-dk)*(2*(djP-dkP)*(djP-dkP)/(dSum*dSum*dSum) - (djPP+dkPP)/(dSum*dSum));
        return (djPP-dkPP)/dSum - 2*(djP-dkP)*(djP+dkP)/(dSum*dSum) + (dj-dk)*(2*(djP+dkP)*(djP+dkP)/(dSum*dSum*dSum) - (djPP+dkPP)/(dSum*dSum));
    }

    @Override
    public int size() {
        return ds.size();
    }
}
