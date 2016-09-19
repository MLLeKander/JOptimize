package nl.rug.joptimize.learn.gmlvq;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.AbstractSeparableCostFunction;

public class GMLVQCostFunction extends AbstractSeparableCostFunction<GMLVQOptParam> {
    private final LabeledDataSet ds;

    public GMLVQCostFunction(LabeledDataSet ds) {
        this.ds = ds;
    }

    @Override
    public double error(GMLVQOptParam params, int exampleNdx) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);
        double dj = params.dist(j, data), dk = params.dist(k, data);
        return (dj - dk) / (dk + dj);
    }

    @Override
    public GMLVQOptParam deriv(GMLVQOptParam params, int exampleNdx, GMLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        if (j < 0 || k < 0) System.out.println("WTF is happening? "+params);
        
        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[][] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);
        //double dSum = dj+dk;

        int dims = params.dimensions(), numWeights = params.numWeights();
        
        //TODO: This can probably be sped up by not using the firstDerivLVQ calls...
        double[] dj_pjP = new double[dims], dk_pkP = new double[dims];
        for (int i = 0; i < numWeights; i++) {
            double compJ = 0, compK = 0;
            double[] wRow = weights[i], owRow = oweights[i];
            for (int o = 0; o < dims; o++) {
                compJ += wRow[o]*(pj[o] - data[o]);
                compK += wRow[o]*(pk[o] - data[o]);
            }
            for (int o = 0; o < dims; o++) {
                double dj_wP = 2*compJ*(pj[o] - data[o]);
                double dk_wP = 2*compK*(pk[o] - data[o]);
                owRow[o] += firstDerivLVQ(dj, dk, dj_wP, dk_wP);
                dj_pjP[o] += compJ*wRow[o];
                dk_pkP[o] += compK*wRow[o];
            }
        }
        for (int o = 0; o < dims; o++) {
            opj[o] += firstDerivLVQ(dj, dk, 2*dj_pjP[o], 0);
            opk[o] += firstDerivLVQ(dj, dk, 0, 2*dk_pkP[o]);
        }

        return out;
    }

    @Override
    public GMLVQOptParam hesseDiag(GMLVQOptParam params, int exampleNdx, GMLVQOptParam out) {
        double[] data = ds.getData(exampleNdx);
        int label = ds.getLabel(exampleNdx);
        // J = same, K = different
        int j = params.getClosestCorrectProtoNdx(data, label, exampleNdx);
        int k = params.getClosestIncorrectProtoNdx(data, label, exampleNdx);

        double[] pj = params.prototypes[j], pk = params.prototypes[k];
        double[] opj = out.prototypes[j], opk = out.prototypes[k];
        double[][] weights = params.weights, oweights = out.weights;
        double dj = params.dist(j, data), dk = params.dist(k, data);

        int dims = params.dimensions(), numWeights = params.numWeights();

        //TODO: This can probably be sped up by not using the secondDerivLVQ calls...
        double[] dj_pjP = new double[dims], dj_pjPP = new double[dims];
        double[] dk_pkP = new double[dims], dk_pkPP = new double[dims];
        for (int i = 0; i < numWeights; i++) {
            double[] wRow = weights[i], owRow = oweights[i];
            
            double compJ = 0, compK = 0;
            for (int o = 0; o < dims; o++) {
                compJ += wRow[o]*(pj[o] - data[o]);
                compK += wRow[o]*(pk[o] - data[o]);
            }
            for (int o = 0; o < dims; o++) {
                double deltaJ = pj[o] - data[o], deltaK = pk[o] - data[o];
                double w = wRow[o];
                double dj_wP = 2*compJ*deltaJ, dj_wPP = 2*deltaJ*deltaJ;
                double dk_wP = 2*compK*deltaK, dk_wPP = 2*deltaK*deltaK;
                owRow[o] += secondDerivLVQ(dj, dk, dj_wP, dk_wP, dj_wPP, dk_wPP);
                dj_pjP[o] += compJ*w;
                dk_pkP[o] += compK*w;
                dj_pjPP[o] += w*w;
                dk_pkPP[o] += w*w;
            }
        }
        for (int o = 0; o < dims; o++) {
            opj[o] += secondDerivLVQ(dj, dk, 2*dj_pjP[o], 0, 2*dj_pjPP[o], 0);
            opk[o] += secondDerivLVQ(dj, dk, 0, 2*dk_pkP[o], 0, 2*dk_pkPP[o]);
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
