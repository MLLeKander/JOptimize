// Lots of redundant code to ensure premature optimization.

package nl.rug.joptimize.learn.glvq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import nl.rug.joptimize.learn.LabeledDataSet;
import nl.rug.joptimize.opt.OptParam;

public class GLVQOptParam implements OptParam<GLVQOptParam> {
    double[][] prototypes;
    int[] labels;
    // TODO Holy shit this is bad.
    static ArrayList<Integer> cMemo = new ArrayList<Integer>();
    static ArrayList<Integer> iMemo = new ArrayList<Integer>();

    public GLVQOptParam(double[][] prototypes, int[] labels, int dsSize) {
        this.prototypes = prototypes;
        this.labels = labels;
        initMemos(dsSize);
    }

    public GLVQOptParam(int classes, int dimensions, int datasetSize) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, dimensions, datasetSize);
    }

    public GLVQOptParam(LabeledDataSet ds) {
        this(ds.classes(), ds.dimensions(), ds.size());
    }

    public GLVQOptParam(int prototypesPerClass, int classes, int dimensions, int dsSize) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, prototypesPerClass);
        init(ppc, dimensions, dsSize);
    }

    public GLVQOptParam(int prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.classes(), ds.dimensions(), ds.size());
    }

    public GLVQOptParam(int[] prototypesPerClass, int dimensions, int dsSize) {
        init(prototypesPerClass, dimensions, dsSize);
    }

    public GLVQOptParam(int[] prototypesPerClass, LabeledDataSet ds) {
        this(prototypesPerClass, ds.dimensions(), ds.size());
        assert (ds.classes() == prototypesPerClass.length);
    }

    private void init(int[] ppc, int dimensions, int dsSize) {
        int protoCount = 0;
        for (int i : ppc) {
            protoCount += i;
        }
        
        this.prototypes = new double[protoCount][dimensions];
        for (double[] row : prototypes) {
            for (int i = 0; i < row.length; i++) {
                row[i] += (Math.random()-0.5)/100;
            }
        }
        
        this.labels = new int[protoCount];
        for (int i = 0, ndx = 0; i < ppc.length; i++) {
            for (int j = 0; j < ppc[i]; j++) {
                this.labels[ndx++] = i;
            }
        }
        
        initMemos(dsSize);
    }

    private void initMemos(int dsSize) {
        // TODO BAD
        if (cMemo.size() != dsSize) {
            cMemo = new ArrayList<Integer>(Collections.nCopies(dsSize, -1));
            iMemo = new ArrayList<Integer>(Collections.nCopies(dsSize, -1));
        }
    }

    public void setProto(int ndx, double[] data) {
        assert (data.length == this.prototypes[ndx].length);
        this.prototypes[ndx] = data;
    }

    public int getClosestProtoNdx(double[] data) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            double currDist = dist(i, data);//, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestProtoLabel(double[] data) {
        return this.labels[this.getClosestProtoNdx(data)];
    }

    public double[] getClosestProto(double[] data) {
        return this.prototypes[this.getClosestProtoNdx(data)];
    }

    public int getClosestCorrectProtoNdx(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label) {
                continue;
            }

            double currDist = dist(i, data);//, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestCorrectProtoNdx(double[] data, int label, int exNdx) {
        //System.err.println(Arrays.toString(data));
        //System.err.println(Arrays.toString(this.prototypes[0]));
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = cMemo.get(exNdx), minProto = minNdx;
        double minDist = minNdx == -1 ? Double.MAX_VALUE : dist(minNdx, data);
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label || i == minProto) {
                continue;
            }

            double currDist = dist(i, data);//, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        cMemo.set(exNdx, minNdx);
        return minNdx;
    }

    public double[] getClosestCorrectProto(double[] data, int label) {
        return this.prototypes[this.getClosestCorrectProtoNdx(data, label)];
    }

    public int getClosestIncorrectProtoNdx(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < protos; i++) {
            if (labels[i] == label) {
                continue;
            }

            double currDist = dist(i, data);//, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int getClosestIncorrectProtoNdx(double[] data, int label, int exNdx) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos();
        int minNdx = iMemo.get(exNdx), minProto = minNdx;
        double minDist = minNdx == -1 ? Double.MAX_VALUE : dist(minNdx, data);
        for (int i = 0; i < protos; i++) {
            if (labels[i] == label || i == minProto) {
                continue;
            }

            double currDist = dist(i, data, minDist);
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        iMemo.set(exNdx, minNdx);
        return minNdx;
    }

    public double[] getClosestIncorrectProto(double[] data, int label) {
        return this.prototypes[this.getClosestIncorrectProtoNdx(data, label)];
    }

    public int getLabel(int ndx) {
        return this.labels[ndx];
    }

    public int numProtos() {
        return this.prototypes.length;
    }

    public int dimensions() {
        return this.prototypes[0].length;
    }

    @Override
    public GLVQOptParam add(GLVQOptParam o) {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] + o.prototypes[i][j];
            }
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam add_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam sub(GLVQOptParam o) {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] - o.prototypes[i][j];
            }
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam sub_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] -= o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam zero() {
        return new GLVQOptParam(newProtos(), this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam zero_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 0d);
        }
        return this;
    }

    @Override
    public GLVQOptParam one() {
        double[][] newProtos = newProtos();
        for (double[] proto : newProtos) {
            Arrays.fill(proto, 1d);
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam one_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 1d);
        }
        return this;
    }

    @Override
    public GLVQOptParam dotprod(GLVQOptParam o) {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * o.prototypes[i][j];
            }
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam dotprod_s(GLVQOptParam o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam multiply(double o) {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * o;
            }
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam multiply_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o;
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam inv() {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = 1 / this.prototypes[i][j];
            }
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public GLVQOptParam inv_s() {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] = 1 / this.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public GLVQOptParam copy() {
        double[][] newProtos = this.newProtos();
        int dims = this.dimensions();
        for (int i = 0; i < newProtos.length; i++) {
            System.arraycopy(this.prototypes[i], 0, newProtos[i], 0, dims);
        }
        return new GLVQOptParam(newProtos, this.labels, cMemo.size());
    }

    @Override
    public double squaredNorm() {
        double out = 0;
        for (double[] row : this.prototypes) {
            for (double d : row) {
                out += d*d;
            }
        }
        return out;
    }

    private double[][] newProtos() {
        return new double[this.numProtos()][this.dimensions()];
    }

    public double dist(int ndx, double[] data) {
        double[] proto = this.prototypes[ndx];
        assert (proto.length == data.length);
        double out = 0;
        for (int i = 0; i < data.length; i++) {
            double tmp = proto[i] - data[i];
            out += tmp * tmp;
        }
        return out;
    }

    private double dist(int ndx, double[] data, double minDist) {
        double[] proto = this.prototypes[ndx];
        double out = 0;
        for (int i = 0; i < data.length; i++) {
            double tmp = proto[i] - data[i];
            out += tmp * tmp;
            if (out > minDist) {
                break;
            }
        }
        return out;
    }

    public String toString() {
        return Arrays.deepToString(prototypes);
    }
}
