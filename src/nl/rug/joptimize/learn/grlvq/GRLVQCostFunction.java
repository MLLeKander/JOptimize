package nl.rug.joptimize.learn.grlvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.AbstractSeparableCostFunction;

public class GRLVQCostFunction extends AbstractSeparableCostFunction<GRLVQOptParam> {
    private final LabeledDataSet ds;

    public GRLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
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
    public GRLVQOptParam deriv(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
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

        int dims = params.dimensions();

        for (int ndx = 0; ndx < dims; ndx++) {
            double deltaJ = pj[ndx] - data[ndx];
            double deltaK = pk[ndx] - data[ndx];
            double w = weights[ndx];

            double dj_pjP = 2*w*w*deltaJ;
            double dk_pkP = 2*w*w*deltaK;
            double dk_pjP = 0, dj_pkP = 0;
            opj[ndx] += firstDerivLVQ(dj, dk, dj_pjP, dk_pjP);
            opk[ndx] += firstDerivLVQ(dj, dk, dj_pkP, dk_pkP);
            double dj_wP = 2*w*deltaJ*deltaJ;
            double dk_wP = 2*w*deltaK*deltaK;
            oweights[ndx] += firstDerivLVQ(dj, dk, dj_wP, dk_wP);
        }

        return out;
    }

    @Override
    public GRLVQOptParam hesseDiag(GRLVQOptParam params, int exampleNdx, GRLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);

        int dims = params.dimensions();

        for (int ndx = 0; ndx < dims; ndx++) {
        	double w = weights[ndx];
            double diffJ = pj[ndx] - data[ndx], diffK = pk[ndx] - data[ndx];
            
            double dj_pjP = 2*w*w*diffJ, dj_pjPP = 2*w*w;
            double dk_pkP = 2*w*w*diffK, dk_pkPP = 2*w*w;
            double dk_pjP = 0, dj_pkP = 0;
            double dk_pjPP = 0, dj_pkPP = 0;
            opj[ndx] += secondDerivLVQ(dj, dk, dj_pjP, dk_pjP, dj_pjPP, dk_pjPP);
            opk[ndx] += secondDerivLVQ(dj, dk, dj_pkP, dk_pkP, dj_pkPP, dk_pkPP);

            double dj_wP = 2*w*diffJ*diffJ, dk_wP = 2*w*diffK*diffK;
            double dj_wPP = 2*diffJ*diffJ, dk_wPP = 2*diffK*diffK;
            oweights[ndx] += secondDerivLVQ(dj, dk, dj_wP, dk_wP, dj_wPP, dk_wPP);
        }

        return out;
    }
    
    private static double firstDerivLVQ(double dj, double dk, double djP, double dkP) {
        double dSum = dj+dk;
        return 2*(dk*djP - dj*dkP)/(dSum*dSum);
    }
    
    private static double secondDerivLVQ(double dj, double dk, double djP, double dkP, double djPP, double dkPP) {
        double dSum = dj+dk;
        return (djPP-dkPP)/dSum - 2*(djP-dkP)*(djP+dkP)/(dSum*dSum) + (dj-dk)*(2*(djP+dkP)*(djP+dkP)/(dSum*dSum*dSum) - (djPP+dkPP)/(dSum*dSum));
    }

    @Override
    public int size() {
        return ds.size();
    }
}
