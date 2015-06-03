// Lots of redundant code to ensure premature optimization.

package nl.rug.joptimize.learn.lvq;

import java.util.Arrays;

import nl.rug.joptimize.opt.OptParam;

public class LVQOptParam implements OptParam {
    double[][] prototypes;
    int[] labels;

    public LVQOptParam(double[][] prototypes, int[] labels) {
        this.prototypes = prototypes;
        this.labels = labels;
    }

    public LVQOptParam(int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, 1);
        init(ppc, dimensions);
    }

    public LVQOptParam(int prototypesPerClass, int classes, int dimensions) {
        int[] ppc = new int[classes];
        Arrays.fill(ppc, prototypesPerClass);
        init(ppc, dimensions);
    }

    public LVQOptParam(int[] prototypesPerClass, int dimensions) {
        init(prototypesPerClass, dimensions);
    }

    private void init(int[] ppc, int dimensions) {
        int protoCount = 0;
        for (int i : ppc) {
            protoCount += i;
        }
        this.prototypes = new double[protoCount][dimensions];
        this.labels = new int[protoCount];
        for (int i = 0, ndx = 0; i < ppc.length; i++) {
            for (int j = 0; j < ppc[i]; j++) {
                this.labels[ndx++] = i;
            }
        }
    }

    public void setProto(int ndx, double[] data) {
        assert (data.length == this.prototypes[ndx].length);
        this.prototypes[ndx] = data;
    }
    
    public int getClosestCorrectProto(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos(), dims = this.dimensions();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;  
        for (int i = 0; i < protos; i++) {
            if (labels[i] != label) {
                continue;
            }
            
            double[] proto = this.prototypes[i];
            
            double currDist = 0;
            for (int j = 0; j < dims; j++) {
                double tmp = data[j]-proto[j];
                currDist += tmp*tmp;
                
                if (currDist > minDist) {
                    break;
                }
            }
            
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }
    
    public int getClosestIncorrectProto(double[] data, int label) {
        assert (data.length == this.prototypes[0].length);
        int protos = this.numProtos(), dims = this.dimensions();
        int minNdx = -1;
        double minDist = Double.MAX_VALUE;  
        for (int i = 0; i < protos; i++) {
            if (labels[i] == label) {
                continue;
            }
            
            double[] proto = this.prototypes[i];
            
            double currDist = 0;
            for (int j = 0; j < dims; j++) {
                double tmp = data[j]-proto[j];
                currDist += tmp*tmp;
                
                if (currDist > minDist) {
                    break;
                }
            }
            
            if (currDist < minDist) {
                minNdx = i;
                minDist = currDist;
            }
        }
        return minNdx;
    }

    public int numProtos() {
        return this.prototypes.length;
    }

    public int dimensions() {
        return this.prototypes[0].length;
    }

    @Override
    public OptParam add(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] + lvqo.prototypes[i][j];
            }
        }
        return new LVQOptParam(newProtos, this.labels);
    }

    @Override
    public OptParam add_s(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] += lvqo.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public OptParam sub(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] - lvqo.prototypes[i][j];
            }
        }
        return new LVQOptParam(newProtos, this.labels);
    }

    @Override
    public OptParam sub_s(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] -= lvqo.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public OptParam zero() {
        return new LVQOptParam(newProtos(), this.labels);
    }

    @Override
    public OptParam zero_s() {
        for (double[] proto : this.prototypes) {
            Arrays.fill(proto, 0d);
        }
        return this;
    }

    @Override
    public OptParam dotprod(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * lvqo.prototypes[i][j];
            }
        }
        return new LVQOptParam(newProtos, this.labels);
    }

    @Override
    public OptParam dotprod_s(OptParam o) {
        LVQOptParam lvqo = (LVQOptParam) o;
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= lvqo.prototypes[i][j];
            }
        }
        return this;
    }

    @Override
    public OptParam multiply(double o) {
        double[][] newProtos = newProtos();
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                newProtos[i][j] = this.prototypes[i][j] * o;
            }
        }
        return new LVQOptParam(newProtos, this.labels);
    }

    @Override
    public OptParam multiply_s(double o) {
        int protos = this.numProtos(), dims = this.dimensions();
        for (int i = 0; i < protos; i++) {
            for (int j = 0; j < dims; j++) {
                this.prototypes[i][j] *= o;
            }
        }
        return this;
    }
    
    private double[][] newProtos() {
        return new double[this.numProtos()][this.dimensions()];
    }
}
